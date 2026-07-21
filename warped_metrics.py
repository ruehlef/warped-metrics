"""Shared helpers for the warped-metrics notebooks.

This module collects the notebook-local model wrappers, point generation,
and WP-metric utilities into one importable place.
"""

from __future__ import annotations

import decimal
import itertools as it
import math
import os
import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
from sympy import LeviCivita
from tqdm import tqdm

from cymetric.tensorflow import KaehlerCallback, RicciCallback, SigmaCallback, SigmaLoss, VolkCallback
try:
    from cymetric.tensorflow.models.helper import prepare_basis as prepare_tf_basis, train_model
except ImportError:
    from cymetric.models.tfhelper import prepare_tf_basis, train_model
from cymetric.tensorflow import PhiFSModel
from cymetric.pointgen.pointgen_mathematica import PointGeneratorMathematica
from wolframclient.deserializers import WXFConsumer, binary_deserialize
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import Global as wlGlobal
from wolframclient.language import wl, wlexpr
from wolframclient.serializers import export as wlexport


def configure_tensorflow_runtime(*, cpu_only: bool = True, intra_threads: int = 8, inter_threads: int = 2) -> None:
    """Apply notebook-friendly TensorFlow runtime defaults.

    The function is safe to call after importing TensorFlow.
    """

    tf.get_logger().setLevel("ERROR")

    try:
        tf.config.threading.set_intra_op_parallelism_threads(intra_threads)
        tf.config.threading.set_inter_op_parallelism_threads(inter_threads)
    except RuntimeError:
        # Threading can only be configured before TensorFlow initializes devices.
        pass

    if cpu_only:
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass


class ComplexFunctionConsumer(WXFConsumer):
    """Map Wolfram Language `Complex` objects to Python complex numbers."""

    _bigreal_re = re.compile(r"([^`]+)(`[0-9.]+){0,1}(\*\^[0-9]+){0,1}")

    def build_function(self, head, args, **kwargs):
        if head == wl.Complex and len(args) == 2:
            return complex(*args)
        return super().build_function(head, args, **kwargs)

    def consume_bigreal(self, current_token, tokens, **kwargs):
        match = self._bigreal_re.match(current_token.data)
        if match:
            num, _prec, exp = match.groups()
            if exp:
                return decimal.Decimal(f"{num}e{exp[2:]}")
            return complex(num)

        raise ValueError(f"Invalid big real value: {current_token.data}")


class PointGeneratorMathematicaIPS(PointGeneratorMathematica):
    """Point generator that loads an IPS-specific Mathematica helper script."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kappas = None

    @staticmethod
    def _setup_session(mathematica_session, script_path: Optional[Path] = None) -> None:
        mathematica_session.evaluate(wlexpr("ClientLibrary`SetErrorLogLevel[]"))

        if script_path is None:
            script_path = Path(__file__).with_name("PointGeneratorMathematicaIPS.m")

        if not script_path.exists():
            raise FileNotFoundError(
                f"Mathematica helper script not found: {script_path}. "
                "Place PointGeneratorMathematicaIPS.m next to warped_metrics.py or pass script_path."
            )

        mathematica_session.evaluate(wl.Get(str(script_path.resolve())))

    def generate_points(self, n_p, num_regions=5, psi=0.0 + 0.0j, rejection_sampling=True, nproc=-1, **kwargs):
        script_path = kwargs.pop("script_path", None)
        with WolframLanguageSession(kernel_loglevel=self.level, kernel=self.kernel_path) as mathematica_session:
            self.wl_session = mathematica_session
            self._setup_session(mathematica_session, script_path=script_path)
            self._start_parallel_kernels(mathematica_session, nproc)

            get_points_mathematica = mathematica_session.function(wlGlobal.GeneratePointsMIPS)
            pts = get_points_mathematica(n_p, num_regions, psi, self.nfold, rejection_sampling)
            pts = wlexport(pts, target_format="wxf")
            pts = binary_deserialize(pts, consumer=ComplexFunctionConsumer())

        data_types = [
            ("point", np.complex128, self.ncoords),
            ("weight", np.float64),
            ("omega", np.float64),
        ]
        dtype = np.dtype(data_types)

        self.kappas = np.array(pts[3], dtype=float)
        self.selected_t = np.array(pts[4], dtype=int)

        point_weights = np.zeros((n_p,), dtype=dtype)
        point_weights["point"], point_weights["weight"], point_weights["omega"] = pts[0], pts[1], pts[2]
        return np.array(point_weights)

    def prepare_dataset(
        self,
        n_p,
        dirname,
        num_regions=5,
        psi=0.0 + 0.0j,
        rejection_sampling=True,
        train_test_split=0.9,
        **kwargs,
    ):
        dirname = Path(dirname)
        dirname.mkdir(parents=True, exist_ok=True)

        point_weights_omega = self.generate_points(
            n_p,
            num_regions=num_regions,
            psi=psi,
            rejection_sampling=rejection_sampling,
            **kwargs,
        )
        points = point_weights_omega["point"]
        weights = np.expand_dims(point_weights_omega["weight"], -1)
        omega = np.expand_dims(point_weights_omega["omega"], -1)

        t_i = int(n_p * train_test_split)
        X_train = np.concatenate((points[:t_i].real, points[:t_i].imag), axis=-1)
        y_train = np.concatenate((weights[:t_i], omega[:t_i]), axis=1)
        X_val = np.concatenate((points[t_i:].real, points[t_i:].imag), axis=-1)
        y_val = np.concatenate((weights[t_i:], omega[t_i:]), axis=1)
        val_pullbacks = self.pullbacks(points[t_i:])

        np.savez_compressed(
            dirname / "dataset",
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            val_pullbacks=val_pullbacks,
        )
        return self.kappas


def pullbacks(pointgen, points, j_elim=None, ncoords=None, n_fold=None, nhyper=None, basis=None):
    """Compute pullbacks for a batch of points.

    The function accepts explicit dimensions, but can infer them from the point
    generator and basis when they are not supplied.
    """

    points = tf.convert_to_tensor(points)
    basis = basis if basis is not None else getattr(pointgen, "BASIS", None)

    if ncoords is None:
        ncoords = int(getattr(pointgen, "ncoords", points.shape[-1] // 2))
    if n_fold is None:
        n_fold = int(getattr(pointgen, "nfold", ncoords - 1))
    if nhyper is None:
        if basis is not None and "NHYPER" in basis:
            nhyper = int(np.real(basis["NHYPER"]))
        else:
            nhyper = int(getattr(pointgen, "nhyper", 1))

    if j_elim is None:
        dQdz_indices = pointgen._find_max_dQ_coords(points)
        full_mask = tf.cast(tf.math.abs(points - tf.complex(1.0, 0.0)) > 1e-8, dtype=tf.float32)
    else:
        dQdz_indices = j_elim
        full_mask = tf.ones((tf.shape(points)[0], ncoords), dtype=tf.float32)
        dQdz_indices = tf.cast(dQdz_indices, tf.int32)
        if dQdz_indices.shape.rank == 1:
            dQdz_indices = dQdz_indices[:, None]
        for i in range(nhyper):
            full_mask -= tf.one_hot(dQdz_indices[:, i], ncoords, dtype=tf.float32)

    dQdz_indices = tf.cast(dQdz_indices, tf.int64)

    for i in range(nhyper):
        dQdz_mask = -tf.one_hot(dQdz_indices[:, i], ncoords, dtype=tf.float32)
        full_mask = tf.math.add(full_mask, dQdz_mask)

    n_p = tf.shape(full_mask)[0]
    full_mask = tf.cast(full_mask, dtype=tf.bool)
    x_z_indices = tf.cast(tf.where(full_mask), dtype=tf.int64)
    good_indices = tf.cast(x_z_indices[:, 1:2], dtype=tf.int64)

    pullbacks_tensor = tf.zeros((n_p, n_fold, ncoords), dtype=tf.complex64)

    y_indices = tf.repeat(tf.expand_dims(tf.cast(tf.range(n_fold), dtype=tf.int64), 0), n_p, axis=0)
    y_indices = tf.reshape(y_indices, (-1, 1))
    diag_indices = tf.concat((x_z_indices[:, 0:1], y_indices, good_indices), axis=-1)
    pullbacks_tensor = tf.tensor_scatter_nd_update(
        pullbacks_tensor,
        diag_indices,
        tf.ones(n_fold * n_p, dtype=tf.complex64),
    )

    fixed_indices = tf.reshape(dQdz_indices, (-1, 1))
    for i in range(nhyper):
        pia_polys = tf.gather_nd(basis[f"DQDZB{i}"], good_indices)
        pia_factors = tf.gather_nd(basis[f"DQDZF{i}"], good_indices)

        cpoints = tf.expand_dims(tf.repeat(points, n_fold, axis=0), 1)
        pia = tf.math.pow(cpoints, pia_polys)
        pia = tf.reduce_prod(pia, axis=-1)
        pia = tf.reduce_sum(tf.multiply(pia_factors, pia), axis=-1)
        pia = tf.reshape(pia, (-1, 1, n_fold))

        if i == 0:
            dz_hyper = pia
        else:
            dz_hyper = tf.concat((dz_hyper, pia), axis=1)

        pif_polys = tf.gather_nd(basis[f"DQDZB{i}"], fixed_indices)
        pif_factors = tf.gather_nd(basis[f"DQDZF{i}"], fixed_indices)
        pif = tf.expand_dims(tf.repeat(points, nhyper, axis=0), 1)
        pif = tf.math.pow(pif, pif_polys)
        pif = tf.reduce_prod(pif, axis=-1)
        pif = tf.reduce_sum(tf.multiply(pif_factors, pif), axis=-1)
        pif = tf.reshape(pif, (-1, 1, nhyper))

        if i == 0:
            b_matrix = pif
        else:
            b_matrix = tf.concat((b_matrix, pif), axis=1)

    all_dzdz = tf.einsum("xij,xjk->xki", tf.linalg.inv(b_matrix), tf.complex(-1.0, 0.0) * dz_hyper)

    for i in range(nhyper):
        fixed_indices_i = tf.reshape(tf.repeat(dQdz_indices[:, i], n_fold), (-1, 1))
        fixed_indices_i = tf.cast(fixed_indices_i, dtype=tf.int64)
        zjzi_indices = tf.concat((x_z_indices[:, 0:1], y_indices, fixed_indices_i), axis=-1)
        zjzi_values = tf.reshape(all_dzdz[:, :, i], [n_fold * n_p])
        pullbacks_tensor = tf.tensor_scatter_nd_update(pullbacks_tensor, zjzi_indices, zjzi_values)

    return pullbacks_tensor


def trace_after_row_insertion(d_theta_z, patch_idx, ncoords):
    """Trace of the matrix after inserting a zero row at `patch_idx`.

    d_theta_z has shape (N, ncoords-1, ncoords).
    """

    d_theta_z = tf.convert_to_tensor(d_theta_z)
    patch_idx = tf.cast(patch_idx, tf.int32)

    n_batch = tf.shape(d_theta_z)[0]
    i = tf.tile(tf.range(ncoords, dtype=tf.int32)[None, :], [n_batch, 1])

    row = i - tf.cast(i > patch_idx[:, None], tf.int32)
    col = i
    valid = tf.not_equal(i, patch_idx[:, None])

    row_v = tf.boolean_mask(row, valid)
    col_v = tf.boolean_mask(col, valid)
    batch_ids = tf.repeat(tf.range(n_batch, dtype=tf.int32), repeats=ncoords - 1)

    idx = tf.stack([batch_ids, row_v, col_v], axis=1)
    diag_vals = tf.gather_nd(d_theta_z, idx)
    return tf.reduce_sum(tf.reshape(diag_vals, [n_batch, ncoords - 1]), axis=1)


def compute_weil_petersson(
    d_thetas: tf.Tensor,
    pb: tf.Tensor,
    omega: tf.Tensor,
    weights: tf.Tensor,
    max_dQ_coords: tf.Tensor,
    points: Optional[tf.Tensor] = None,
    patch_idx: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """Compute the sampled Weil-Petersson scalar.

    Either `patch_idx` or `points` must be provided to determine the active patch.
    """

    del omega

    d_thetas = tf.convert_to_tensor(d_thetas)
    pb = tf.cast(pb, tf.complex64)
    weights = tf.cast(weights, tf.complex64)

    last = tf.shape(d_thetas)[-1]
    half = last // 2
    d_tr_dr = d_thetas[..., 0, :half]
    d_ti_dr = d_thetas[..., 1, :half]
    d_tr_di = d_thetas[..., 0, half:]
    d_ti_di = d_thetas[..., 1, half:]

    d_theta_z = 0.5 * tf.complex(d_tr_dr + d_ti_di, d_ti_dr - d_tr_di)
    d_theta_bz = 0.5 * tf.complex(d_tr_dr - d_ti_di, d_ti_dr + d_tr_di)
    d_theta_z = tf.cast(d_theta_z, tf.complex64)
    d_theta_bz = tf.cast(d_theta_bz, tf.complex64)

    if patch_idx is None:
        if points is None:
            raise ValueError("Either `points` or `patch_idx` must be provided.")
        is_one = tf.math.abs(points - tf.complex(1.0, 0.0)) < 1e-8
        patch_idx = tf.argmax(tf.cast(is_one, tf.int32), axis=1, output_type=tf.int32)

    patch_idx = tf.cast(patch_idx, tf.int32)
    max_dQ_coords = tf.cast(max_dQ_coords, tf.int32)
    index = max_dQ_coords - tf.cast(max_dQ_coords > patch_idx, tf.int32)

    n_batch = tf.shape(d_theta_bz)[0]
    n_rows = tf.shape(d_theta_bz)[1]
    rows = tf.tile(tf.range(n_rows, dtype=tf.int32)[None, :], [n_batch, 1])
    keep = tf.not_equal(rows, index[:, None])
    gather_rows = tf.reshape(tf.boolean_mask(rows, keep), [n_batch, n_rows - 1])

    dbthetas = tf.gather(d_theta_bz, gather_rows, batch_dims=1, axis=1)
    dbthetas = tf.cast(dbthetas, tf.complex64)

    bpbs = tf.cast(tf.math.conj(pb), tf.complex64)
    dbthetas_pb = tf.einsum("bij,bkj->bik", dbthetas, bpbs)

    signs = tf.cast([(-1.0) ** (i + 1) for i in range(dbthetas_pb.shape[1])], dbthetas_pb.dtype)
    p21 = dbthetas_pb * tf.reshape(signs, (-1, 1))

    shape_mat = (dbthetas_pb.shape[1], dbthetas_pb.shape[2])
    ii, jj = tf.meshgrid(tf.range(shape_mat[0]), tf.range(shape_mat[1]), indexing="ij")
    pattern_tensor = tf.cast(tf.pow(-1, ii + jj + 1), tf.complex64)
    res = tf.einsum("ij,xij,xji->x", pattern_tensor, p21, tf.math.conj(p21))

    trace_full = trace_after_row_insertion(d_theta_z, patch_idx, dbthetas_pb.shape[1] + 2)
    p30 = -tf.cast(trace_full, tf.complex64)
    p30squared = p30 * tf.math.conj(p30)

    omegaSquared = tf.cast(tf.reduce_sum(weights), tf.complex64)
    dnubnu = tf.reduce_sum(p30 * weights)
    dnubdnu = tf.reduce_sum(res * weights) + tf.reduce_sum(p30squared * weights)

    return -dnubdnu / omegaSquared + (dnubnu * tf.math.conj(dnubnu)) / (omegaSquared ** 2)


def build_phi_network(n_in, nlayer, nHidden, act, n_out=1, mixed_precision_safe=True):
    """Construct the dense network used by the Phi models."""

    model = tfk.Sequential()
    model.add(tfk.Input(shape=(n_in,)))
    for _ in range(nlayer):
        model.add(tfk.layers.Dense(nHidden, activation=act))
    if mixed_precision_safe:
        model.add(tfk.layers.Dense(n_out, use_bias=False, dtype="float32"))
    else:
        model.add(tfk.layers.Dense(n_out, use_bias=False))
    return model


# If tracing/time profiling is stable in environment, can enable tf.function
# @tf.function(jit_compile=False)
def calc_derivatives_fast(
    pts,
    DQDZB, DQDZF,                # tensors (numpy or tf) used for dQ/dz
    DDQDZB, DDQDZF,             # tensors for d^2Q/dz^A dz^B
    qI,
    metric_ambient,              # ambient space metric 
    metric_model,                # learned CY metric model, returns complex g_cy (B, n_fold, n_fold)
    pg,                          # PointGenerator (used for pullbacks / _find_max_dQ_coords)
    ncoords,
):
    """
    Fast version of calc_derivatives.
    Inputs:
      pts: (B, 2*ncoords) float32 real coordinates [real,z_imag concatenated]
      DQDZB, DQDZF, DDQDZB, DDQDZF: precomputed bases (tf or numpy arrays)
      qI: complex powers (1d) as tf.complex64 or convertible
      comp_model: has method fubini_study_amb(pts) returning (B,ncoords,ncoords) complex
      metric_model: takes pts (real float32) and returns (B, n_fold, n_fold) complex64
      pg: point generator object used to compute pullbacks(...)
      ncoords: int
    Returns:
      X (B, n_fold, ncoords) complex64,
      d_X (B, n_fold, 2*ncoords, 2*ncoords) real/imag stacked -> matches your usage (real/imag last axis stacked before)
      d_g_cy (B, n_fold, n_fold, 2*ncoords, 2) or matching your stacked layout (we keep same layout as original)
      d_g_inv ...
    Notes:
      - This uses tf ops only inside GradientTape.
      - If you repeatedly compute for the same pts, consider precomputing pullbacks outside and passing them in.
    """

    # ensure tensors
    pts = tf.convert_to_tensor(pts, dtype=tf.float32)
    # If DQDZB etc are numpy, convert once outside to tf to avoid repeated work.
    DQDZB = tf.convert_to_tensor(DQDZB, dtype=tf.complex64)   # shape like (n_terms, ..., ncoords) - keep original dtype layout
    DQDZF = tf.convert_to_tensor(DQDZF, dtype=tf.complex64)
    DDQDZB = tf.convert_to_tensor(DDQDZB, dtype=tf.complex64)
    DDQDZF = tf.convert_to_tensor(DDQDZF, dtype=tf.complex64)
    qI = tf.convert_to_tensor(qI, dtype=tf.complex64)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(pts)

        # complex coordinates (B, ncoords) complex64
        points = tf.complex(pts[:, :ncoords], pts[:, ncoords:])

        # ambient FS metric (B, ncoords, ncoords) complex64
        g_ambient = metric_ambient(pts)
        # if comp_model returns float dtype, cast to complex
        if g_ambient.dtype.is_floating:
            g_ambient = tf.cast(g_ambient, tf.complex64)

        # --- dQdz (B, ncoords) complex64 ---
        p_exp = tf.expand_dims(tf.expand_dims(points, 1), 1)   # (B,1,1,ncoords)
        # DQDZB expected shape broadcastable to (..., ncoords)
        dQdz = tf.math.pow(p_exp, tf.cast(DQDZB, tf.complex64))
        dQdz = tf.reduce_prod(dQdz, axis=-1)
        dQdz = tf.cast(DQDZF, tf.complex64) * dQdz
        dQdz = tf.reduce_sum(dQdz, axis=-1)                   # (B, ncoords)

        # --- ddQdz (B, ncoords, ncoords) complex64 ---
        p_exp2 = tf.expand_dims(tf.expand_dims(tf.expand_dims(points, 1), 1), 1)  # (B,1,1,1,ncoords)
        ddQdz = tf.math.pow(p_exp2, tf.cast(DDQDZB, tf.complex64))
        ddQdz = tf.reduce_prod(ddQdz, axis=-1)
        ddQdz = tf.cast(DDQDZF, tf.complex64) * ddQdz
        ddQdz = tf.reduce_sum(ddQdz, axis=-1)                  # (B, ncoords, ncoords)
        bddQdz = tf.math.conj(ddQdz)

        # --- find patch index j; use exact equality as you requested ---
        is_one = tf.equal(points, tf.complex(1.0, 0.0))        # (B, ncoords)
        j = tf.argmax(tf.cast(is_one, tf.int32), axis=1, output_type=tf.int32)  # (B,)

        # --- build gather indices to remove j ---
        B = tf.shape(points)[0]
        idx_all = tf.tile(tf.range(ncoords, dtype=tf.int32)[None, :], [B, 1])   # (B, ncoords)
        keep = tf.not_equal(idx_all, j[:, None])                                # (B, ncoords)
        gather_idx = tf.reshape(tf.boolean_mask(idx_all, keep), [B, ncoords - 1])  # (B, ncoords-1)

        # --- gather dQdz without j: (B, ncoords-1) ---
        dQdz_del = tf.gather(dQdz, gather_idx, batch_dims=1)

        # --- gather g_ambient with row/col j removed: (B, ncoords-1, ncoords-1) ---
        # first remove rows then columns using the same indices
        g_ambient_row_del = tf.gather(g_ambient, gather_idx, batch_dims=1, axis=1)
        g_ambient_del = tf.gather(g_ambient_row_del, gather_idx, batch_dims=1, axis=2)

        # cast & conj
        dQdz_del = tf.cast(dQdz_del, tf.complex64)
        bdQdz_del = tf.math.conj(dQdz_del)

        # inverse of small matrix (B, ncoords-1, ncoords-1)
        g_inv = tf.linalg.inv(tf.cast(g_ambient_del, tf.complex64))

        # H and its inverse (B,)
        H = tf.einsum('xij,xi,xj->x', g_inv, bdQdz_del, dQdz_del)
        H_inv = 1.0 / H

        # qs (B,)
        qs = tf.math.pow(points, tf.expand_dims(qI, 0))
        qs = -5.0 * tf.reduce_prod(qs, axis=-1)

        # pullbacks (B, ncoords, ncoords') based on pg implementation
        pbs = pullbacks(pg, points, j_elim = None)
        bpbs = tf.math.conj(pbs)

        # push ambient metric into CY patch frame, invert
        g_ambient_pb = tf.einsum('xji,xik,xlk->xjl', pbs, g_ambient, bpbs)
        g_ambient_inv_pb = tf.linalg.inv(tf.cast(g_ambient_pb, tf.complex64))

        bddQdz_pb = tf.einsum('xji,xik,xlk->xjl', bpbs, bddQdz, bpbs)
        X = -tf.einsum('x,xji,xjk,x->xki', H_inv, g_ambient_inv_pb, bddQdz_pb, qs)

        X_real = tf.stack([tf.math.real(X), tf.math.imag(X)], axis=-1)

        # learned CY metric (model passed in)
        g_cy = metric_model(pts)                   # expect complex64 (B, n_fold, n_fold)
        if not g_cy.dtype.is_complex:
            g_cy = tf.cast(g_cy, tf.complex64)
        g_inv_cy = tf.linalg.inv(g_cy)

        g_cy_real = tf.stack([tf.math.real(g_cy), tf.math.imag(g_cy)], axis=-1)
        g_inv_real = tf.stack([tf.math.real(g_inv_cy), tf.math.imag(g_inv_cy)], axis=-1)

    # Now compute batched jacobians (these are the expensive ops)
    d_X = tape.batch_jacobian(X_real, pts)          # shape like your original d_X
    d_g_cy = tape.batch_jacobian(g_cy_real, pts)
    d_g_inv = tape.batch_jacobian(g_inv_real, pts)
    del tape

    # return complex tensors so downstream code uses same conventions
    return X, d_X, d_g_cy, d_g_inv


def holomorphic_volume_form(pointgen, points):
    indices = pointgen._find_max_dQ_coords(points)
    omega = tf.ones_like(points[:, 0])
    for i in range(pointgen.nhyper):
        tmp_omega = tf.math.pow(tf.expand_dims(points, 1), pointgen.BASIS['DQDZB' + str(i)][indices[:, i]])
        tmp_omega = tf.reduce_prod(tmp_omega, axis=-1)
        omega *= tf.reduce_sum(pointgen.BASIS['DQDZF' + str(i)][indices[:, i]] * tmp_omega, axis=-1)
    return 1 / omega


def levi_civita_tensor(dim):   
    arr=np.zeros(tuple([dim for _ in range(dim)]))
    for x in it.permutations(tuple(range(dim))):
        mat = np.zeros((dim, dim), dtype=np.int32)
        for i, j in zip(range(dim), x):
            mat[i, j] = 1
        arr[x]=int(np.linalg.det(mat))
    return arr


class calculate_dQdz:
    def __init__(self, ncoords, monomials, coefficients):
        self.ncoords = ncoords
        self.monomials = monomials
        self.coefficients = coefficients

    def _generate_dQdz_basis(self):
        dQdz_basis = []
        dQdz_factors = []
        for i, m in enumerate(np.eye(self.ncoords, dtype=np.int32)):
            basis = self.monomials - m
            factors = self.monomials[:, i] * self.coefficients
            good = np.ones(len(basis), dtype=bool)
            good[np.where(basis < 0)[0]] = False
            dQdz_basis.append(basis[good])
            dQdz_factors.append(factors[good])
        return dQdz_basis, dQdz_factors


class calculate_ddQdz:
    def __init__(self, ncoords, monomials, coefficients):
        self.ncoords = ncoords
        self.monomials = monomials
        self.coefficients = coefficients

    def calculate_basis(self):
        ddQdz_basis = []
        ddQdz_factors = []
        eye = np.eye(self.ncoords, dtype=np.int32)
        for (i, j), _val in np.ndenumerate(eye):
            m = eye[i, :]
            n = eye[:, j]
            basis = self.monomials - m - n
            factors = self.monomials[:, i] * (self.monomials - m)[:, j] * self.coefficients
            good = np.ones(len(basis), dtype=bool)
            good[np.where(basis < 0)[0]] = False
            ddQdz_basis.append(basis[good])
            ddQdz_factors.append(factors[good])
        return ddQdz_basis, ddQdz_factors


@tf.function
def calc_d_theta_batch(pts, DQDZB, DQDZF, DDQDZB, DDQDZF, qI, ncoords, comp_model, pg=None):
    """Compute one batch of d_theta values for the WP helper cache."""

    del pg

    pts = tf.convert_to_tensor(pts, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(pts)

        points = tf.complex(pts[:, :ncoords], pts[:, ncoords:])
        gfs = comp_model.fubini_study_amb(pts)

        p_exp = tf.expand_dims(tf.expand_dims(points, 1), 1)
        dQdz = tf.math.pow(p_exp, DQDZB)
        dQdz = tf.reduce_prod(dQdz, axis=-1)
        dQdz = DQDZF * dQdz
        dQdz = tf.reduce_sum(dQdz, axis=-1)

        p_exp = tf.expand_dims(tf.expand_dims(tf.expand_dims(points, 1), 1), 1)
        ddQdz = tf.math.pow(p_exp, DDQDZB)
        ddQdz = tf.reduce_prod(ddQdz, axis=-1)
        ddQdz = DDQDZF * ddQdz
        ddQdz = tf.reduce_sum(ddQdz, axis=-1)

        is_one = tf.equal(points, tf.complex(1.0, 0.0))
        j = tf.argmax(tf.cast(is_one, tf.int32), axis=1, output_type=tf.int32)

        batch_size = tf.shape(points)[0]
        idx_all = tf.tile(tf.range(ncoords, dtype=tf.int32)[None, :], [batch_size, 1])
        mask_keep = tf.not_equal(idx_all, j[:, None])
        gather_idx = tf.reshape(tf.boolean_mask(idx_all, mask_keep), [batch_size, ncoords - 1])

        dQdz_del = tf.gather(dQdz, gather_idx, batch_dims=1)
        dQdz_del = tf.cast(dQdz_del, tf.complex64)
        bdQdz_del = tf.math.conj(dQdz_del)

        ddQdz_del = tf.gather(ddQdz, gather_idx, batch_dims=1, axis=2)
        ddQdz_del = tf.cast(ddQdz_del, tf.complex64)
        bddQdz_del = tf.math.conj(ddQdz_del)

        gfs_rows = tf.gather(gfs, gather_idx, batch_dims=1, axis=1)
        gfs_del = tf.gather(gfs_rows, gather_idx, batch_dims=1, axis=2)
        gfs_del = tf.cast(gfs_del, tf.complex64)

        g_inv = tf.linalg.inv(gfs_del)
        H = tf.einsum("bij,bi,bj->b", g_inv, bdQdz_del, dQdz_del)
        H_inv = tf.cast(1.0, H.dtype) / H

        qI = tf.expand_dims(qI, 0)
        qs = tf.math.pow(points, qI)
        qs = -5.0 * tf.reduce_prod(qs, axis=-1)

        theta = -tf.einsum("b,bij,bi,b->bj", H_inv, g_inv, bdQdz_del, qs)
        theta_real = tf.stack([tf.math.real(theta), tf.math.imag(theta)], axis=-1)

    d_theta = tape.batch_jacobian(theta_real, pts, experimental_use_pfor=True)
    return d_theta


def calc_d_theta_par(target_file, pts, DQDZB, DQDZF, DDQDZB, DDQDZF, qI, ncoords, comp_model, pg, batch_size=1000):
    """Cached parallel wrapper for calc_d_theta_batch."""

    if os.path.exists(target_file):
        data_theta = np.load(target_file)
        return tf.cast(data_theta["d_thetas"], dtype=tf.float32)

    pts = tf.convert_to_tensor(pts, dtype=tf.float32)
    n = int(pts.shape[0])
    nb = (n + batch_size - 1) // batch_size

    chunks = []
    for i in tqdm(range(nb)):
        b = pts[i * batch_size:(i + 1) * batch_size]
        chunks.append(calc_d_theta_batch(b, DQDZB, DQDZF, DDQDZB, DDQDZF, qI, ncoords, comp_model, pg))

    d_thetas = tf.concat(chunks, axis=0)
    np.savez(target_file, d_thetas=d_thetas.numpy())
    return tf.cast(d_thetas, dtype=tf.float32)


def calc_d_theta_batch_ambient(pts, DQDZB, DQDZF, DDQDZB, DDQDZF, qI, ncoords, metric_model=None, comp_model=None, pg=None):
    """Ambient wrapper that accepts both metric_model and comp_model names."""

    model = metric_model if metric_model is not None else comp_model
    if model is None:
        raise TypeError("calc_d_theta_batch_ambient requires `metric_model` (or `comp_model`).")
    return calc_d_theta_batch(pts, DQDZB, DQDZF, DDQDZB, DDQDZF, qI, ncoords, model, pg)


def calc_d_theta_par_ambient(
    target_file,
    pts,
    DQDZB,
    DQDZF,
    DDQDZB,
    DDQDZF,
    qI,
    ncoords,
    metric_model=None,
    comp_model=None,
    pg=None,
    batch_size=1000,
):
    """Ambient wrapper that preserves the original notebook keyword API."""

    model = metric_model if metric_model is not None else comp_model
    if model is None:
        raise TypeError("calc_d_theta_par_ambient requires `metric_model` (or `comp_model`).")
    return calc_d_theta_par(target_file, pts, DQDZB, DQDZF, DDQDZB, DDQDZF, qI, ncoords, model, pg, batch_size)


def christoffel(g_model, pts):
    """Compute Christoffel symbols for a complex metric model."""

    pts = tf.convert_to_tensor(pts, dtype=tf.float32)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(pts)
        g = tf.cast(g_model(pts), tf.complex64)
        g_re = tf.math.real(g)
    d_g_re = tape.batch_jacobian(g_re, pts)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(pts)
        g = tf.cast(g_model(pts), tf.complex64)
        g_im = tf.math.imag(g)
    d_g_im = tape.batch_jacobian(g_im, pts)

    ncoords = g_model.ncoords
    dx_g_re = d_g_re[:, :, :, :ncoords]
    dy_g_re = d_g_re[:, :, :, ncoords:]
    dx_g_im = d_g_im[:, :, :, :ncoords]
    dy_g_im = d_g_im[:, :, :, ncoords:]
    d_g = tf.complex(dx_g_re + dy_g_im, -dy_g_re + dx_g_im)

    pbs = g_model.pullbacks(pts)
    d_g_pb = tf.einsum("xbn,xcdn->xbcd", pbs, d_g)
    gamma = tf.einsum("xbcd,xda->xabc", d_g_pb, tf.linalg.inv(g))
    return gamma


def riemann(g_model, pts):
    """Return the Riemann tensor R^a_{b\bar c d}."""

    pts = tf.convert_to_tensor(pts, dtype=tf.float32)
    with tf.GradientTape(persistent=False) as tape:
        tape.watch(pts)
        gamma = christoffel(g_model, pts)
        gamma_re = tf.math.real(gamma)
    d_gamma_re = tape.batch_jacobian(gamma_re, pts)

    with tf.GradientTape(persistent=False) as tape:
        tape.watch(pts)
        gamma = christoffel(g_model, pts)
        gamma_im = tf.math.imag(gamma)
    d_gamma_im = tape.batch_jacobian(gamma_im, pts)

    ncoords = g_model.ncoords
    dx_gamma_re = d_gamma_re[:, :, :, :, :ncoords]
    dy_gamma_re = d_gamma_re[:, :, :, :, ncoords:]
    dx_gamma_im = d_gamma_im[:, :, :, :, :ncoords]
    dy_gamma_im = d_gamma_im[:, :, :, :, ncoords:]
    dbarc_gamma = tf.complex(dx_gamma_re - dy_gamma_im, dy_gamma_re + dx_gamma_im)

    pbs = g_model.pullbacks(pts)
    riem = -tf.einsum("xci,xabdi->xabcd", tf.math.conj(pbs), dbarc_gamma)
    return riem


def compute_riemann(target_file, pts, comp_model, batch_size=10000):
    """Compute and cache the Riemann tensor in a pickle file."""

    pts = tf.cast(pts, dtype=tf.float32)
    num_batches = math.ceil(len(pts) / batch_size)
    if not os.path.exists(target_file):
        for i in tqdm(range(num_batches)):
            batch = pts[i * batch_size:(i + 1) * batch_size]
            if i == 0:
                riem = riemann(comp_model, batch)
            else:
                riem = tf.concat([riem, riemann(comp_model, batch)], axis=0)
        with open(target_file, "wb") as hnd:
            pickle.dump(riem.numpy(), hnd)
    else:
        with open(target_file, "rb") as hnd:
            riem = pickle.load(hnd)
    return tf.cast(riem, dtype=tf.complex64)


def get_chern_classes(riem, comp_model):
    """Return (c1, c2, c3, c3_form) from a sampled Riemann tensor."""

    riem = tf.cast(riem, tf.complex64)
    tr_R = -tf.einsum("xaabc->xcb", riem)
    c1 = 1j / (2 * math.pi) * tr_R

    tr_R2 = tf.einsum("xabmn,xbaop->xnmpo", riem, riem)
    c2 = 1.0 / (2 * (2 * math.pi) ** 2) * (tr_R2 - tf.einsum("xab,xcd->xabcd", tr_R, tr_R))

    tr_R3 = tf.einsum("xabmn,xbcop,xcaqr->xnmporq", riem, riem, riem)
    c3 = (
        1.0 / 3.0 * tf.einsum("xmn,xopqr->xmnopqr", c1, c2)
        + 1.0 / (3 * (2 * math.pi) ** 2) * tf.einsum("xmn,xopqr->xmnopqr", c1, tr_R2)
        - 1j / (3 * (2 * math.pi) ** 3) * tr_R3
    )

    c3_form = 1.0 / math.factorial(comp_model.nfold) * tf.einsum("xmnopqr,moq,npr->x", c3, comp_model.lc, comp_model.lc)
    return c1, c2, c3, c3_form


def integrate_weighted(integrand, pts, wo, comp_model, kappas=[1.], normalize_to_vol=None):
    kappas_t = tf.reshape(tf.cast(tf.convert_to_tensor(kappas), tf.complex64), [-1])
    num_regions = int(kappas_t.shape[0])
    if num_regions <= 0:
        raise ValueError("kappas must contain at least one value")

    num_pts = len(pts)
    num_pts_per_region = num_pts // num_regions

    aux_weights = tf.convert_to_tensor(wo[:, 0] / wo[:, 1], dtype=tf.complex64)
    aux_weights_weighted = []

    for i in range(num_regions):
        start_idx = num_pts_per_region * i
        end_idx = num_pts if i == num_regions - 1 else num_pts_per_region * (i + 1)
        region_weights = aux_weights[start_idx:end_idx] * kappas_t[i]
        aux_weights_weighted.append(region_weights)

    # Concatenate and normalize
    aux_weights_weighted = tf.concat(aux_weights_weighted, axis=0)
    res = (-1.j / 2)**comp_model.nfold * tf.reduce_mean(integrand * aux_weights_weighted, axis=-1)
    #res = tf.reduce_mean(integrand * aux_weights_weighted, axis=-1)
    if normalize_to_vol is not None:
        vol = tf.abs(tf.reduce_mean(tf.linalg.det(comp_model(pts)) * aux_weights_weighted, axis=-1))
        kappa = normalize_to_vol / vol
        vol *= tf.cast(kappa, dtype=vol.dtype)
        res *= tf.cast(kappa, dtype=res.dtype)
    return res


def integrate_standard(integrand, pts, wo, comp_model, normalize_to_vol=None):
    """Integrate a top-form using dataset Monte Carlo weights."""

    aux_weights = tf.convert_to_tensor(wo[:, 0] / wo[:, 1], dtype=tf.complex64)
    aux_weights = tf.repeat(tf.expand_dims(aux_weights, axis=0), repeats=[len(comp_model.BASIS["KMODULI"])], axis=0)

    res = (-1j / 2) ** comp_model.nfold * tf.reduce_mean(integrand * aux_weights, axis=-1)[0]
    if normalize_to_vol is not None:
        vol = tf.abs(tf.reduce_mean(tf.linalg.det(comp_model(pts)) * aux_weights[0], axis=-1))
        kappa = normalize_to_vol / vol
        vol *= tf.cast(kappa, dtype=vol.dtype)
        res *= tf.cast(kappa, dtype=res.dtype)
    return res


def compute_riemann_par_oomsafe(target_file_pickle, pts, comp_model, batch_size=250, also_write_pickle=False):
    """Batch-streamed Riemann computation with a .npy cache for lower peak memory."""

    if os.path.exists(target_file_pickle):
        with open(target_file_pickle, "rb") as hnd:
            arr = pickle.load(hnd)
        return tf.cast(arr, tf.complex64)

    target_npy = os.path.splitext(target_file_pickle)[0] + ".npy"
    if os.path.exists(target_npy):
        arr = np.load(target_npy, mmap_mode="r")
        return tf.cast(arr, tf.complex64)

    pts = np.asarray(pts, dtype=np.float32)
    n = int(pts.shape[0])
    nb = (n + batch_size - 1) // batch_size

    probe_n = min(batch_size, n)
    probe_tf = tf.convert_to_tensor(pts[:probe_n], tf.float32)
    probe_out = riemann(comp_model, probe_tf)
    out_shape = tuple(probe_out.shape[1:])

    mm = np.lib.format.open_memmap(target_npy, mode="w+", dtype=np.complex64, shape=(n,) + out_shape)
    for i in tqdm(range(nb)):
        s = i * batch_size
        e = min((i + 1) * batch_size, n)
        b = tf.convert_to_tensor(pts[s:e], tf.float32)
        mm[s:e] = tf.cast(riemann(comp_model, b), tf.complex64).numpy()
    mm.flush()

    if also_write_pickle:
        with open(target_file_pickle, "wb") as hnd:
            pickle.dump(np.asarray(mm), hnd)

    arr = np.load(target_npy, mmap_mode="r")
    return tf.cast(arr, tf.complex64)


def build_harm_nn(n_in, n_out, layers=3, width=128, act="silu"):
    inp = tfk.Input(shape=(n_in,))
    x = inp
    for _ in range(layers):
        x0 = x
        x = tfk.layers.Dense(width, activation=None)(x)
        x = tfk.layers.LayerNormalization()(x)
        x = tfk.layers.Activation(act)(x)
        # residual if same width
        if x0.shape[-1] == x.shape[-1]:
            x = tfk.layers.Add()([x, x0])
    out = tfk.layers.Dense(n_out, use_bias=False)(x)
    return tfk.Model(inp, out)


def make_dataset_cached_pb(
    x, y,
    *,
    pbs_all,
    g_cy_pb_all, g_inv_pb_all,
    d_gcy_z_pb_all, d_ginv_z_pb_all,
    omega_pb_all, d_omega_z_pb_all,
    batch_size=64,
    shuffle=True,
    shuffle_buf=20000,
):
    """
    All *_all arrays/tensors must be aligned on axis 0 with x and y.
    """
    # Ensure tensors (important if you loaded memmaps)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y)  # keep complex dtype

    pbs_all          = tf.convert_to_tensor(pbs_all, dtype=tf.complex64)
    g_cy_pb_all      = tf.convert_to_tensor(g_cy_pb_all, dtype=tf.complex64)
    g_inv_pb_all     = tf.convert_to_tensor(g_inv_pb_all, dtype=tf.complex64)
    d_gcy_z_pb_all   = tf.convert_to_tensor(d_gcy_z_pb_all, dtype=tf.complex64)
    d_ginv_z_pb_all  = tf.convert_to_tensor(d_ginv_z_pb_all, dtype=tf.complex64)
    omega_pb_all     = tf.convert_to_tensor(omega_pb_all, dtype=tf.complex64)
    d_omega_z_pb_all = tf.convert_to_tensor(d_omega_z_pb_all, dtype=tf.complex64)

    ds = tf.data.Dataset.from_tensor_slices((
        x, y,
        pbs_all,
        g_cy_pb_all, g_inv_pb_all,
        d_gcy_z_pb_all, d_ginv_z_pb_all,
        omega_pb_all, d_omega_z_pb_all
    ))

    if shuffle:
        # buffer size: at most dataset size
        n = tf.shape(x)[0]
        # tf.data needs python int; fall back to shuffle_buf if shape unknown
        try:
            n_int = int(x.shape[0])
            buf = min(shuffle_buf, n_int)
        except Exception:
            buf = shuffle_buf
        ds = ds.shuffle(buf, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    def _pack(xb, yb, pbs, gcy, ginv, dgcyz, dginvz, om, domz):
        cache = {
            "pbs": pbs,
            "g_cy_pb": gcy,
            "g_inv_pb": ginv,
            "d_gcy_z_pb": dgcyz,
            "d_ginv_z_pb": dginvz,
            "omega_pb": om,
            "d_omega_z_pb": domz,
        }
        return xb, (yb, cache)

    return ds.map(_pack, num_parallel_calls=tf.data.AUTOTUNE)


def trace_after_row_insertion(d_theta_z, patch_idx, ncoords):
    """
    d_theta_z: (N, ncoords-1, ncoords) complex
    patch_idx: (N,) int32, position where a zero row would be inserted
    returns: (N,) complex trace of the resulting (ncoords x ncoords) matrix
    without explicitly building it.
    """
    N = tf.shape(d_theta_z)[0]
    i = tf.tile(tf.range(ncoords, dtype=tf.int32)[None, :], [N, 1])          # (N, ncoords)

    # Row index in d_theta_z (shift down by 1 for i > patch_idx)
    row = i - tf.cast(i > patch_idx[:, None], tf.int32)                      # (N, ncoords)
    col = i                                                                  # (N, ncoords)

    # diagonal element at i==patch_idx is zero (inserted row)
    valid = tf.not_equal(i, patch_idx[:, None])                              # (N, ncoords)

    row_v = tf.boolean_mask(row, valid)                                      # (N*(ncoords-1),)
    col_v = tf.boolean_mask(col, valid)                                      # (N*(ncoords-1),)
    batch_ids = tf.repeat(tf.range(N, dtype=tf.int32), repeats=ncoords - 1)  # (N*(ncoords-1),)

    idx = tf.stack([batch_ids, row_v, col_v], axis=1)                        # (N*(ncoords-1), 3)
    diag_vals = tf.gather_nd(d_theta_z, idx)                                 # (N*(ncoords-1),)

    return tf.reduce_sum(tf.reshape(diag_vals, [N, ncoords - 1]), axis=1)


def calc_d_theta_batch(pts, DQDZB, DQDZF, DDQDZB, DDQDZF, qI, ncoords, comp_model, pg=None):
    """
    Vectorized version: no Python loop over points.
    pts: (B, 2*ncoords) real
    Returns: d_theta (B, ?, 2*ncoords) depending on theta_real shape
    """

    pts = tf.convert_to_tensor(pts, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(pts)

        # complex points: (B, ncoords)
        points = tf.complex(pts[:, :ncoords], pts[:, ncoords:])

        # gfs: (B, ncoords, ncoords)
        gfs = comp_model.fubini_study_amb(pts)

        # ---- dQdz ----
        # p_exp: (B, 1, 1, ncoords) so it can broadcast with DQDZB
        p_exp = tf.expand_dims(tf.expand_dims(points, 1), 1)
        dQdz = tf.math.pow(p_exp, DQDZB)
        dQdz = tf.reduce_prod(dQdz, axis=-1)
        dQdz = DQDZF * dQdz
        dQdz = tf.reduce_sum(dQdz, axis=-1)  # (B, ncoords)

        # ---- ddQdz ----
        # p_exp: (B, 1, 1, 1, ncoords)
        p_exp = tf.expand_dims(tf.expand_dims(tf.expand_dims(points, 1), 1), 1)
        ddQdz = tf.math.pow(p_exp, DDQDZB)
        ddQdz = tf.reduce_prod(ddQdz, axis=-1)
        ddQdz = DDQDZF * ddQdz
        ddQdz = tf.reduce_sum(ddQdz, axis=-1)  # (B, ncoords, ncoords)

        is_one = tf.equal(points, tf.complex(1.0, 0.0))   # (B, ncoords)
        j = tf.argmax(tf.cast(is_one, tf.int32), axis=1, output_type=tf.int32)  # (B,), int32

        B = tf.shape(points)[0]
        idx_all = tf.tile(tf.range(ncoords, dtype=tf.int32)[None, :], [B, 1])   # (B, ncoords), int32

        mask_keep = tf.not_equal(idx_all, j[:, None])                           # (B, ncoords)


        # gather_idx: (B, ncoords-1) indices to keep
        gather_idx = tf.reshape(tf.boolean_mask(idx_all, mask_keep), [B, ncoords - 1])

        # ---- delete that coord from dQdz (B, ncoords-1) ----
        dQdz_del = tf.gather(dQdz, gather_idx, batch_dims=1)
        dQdz_del = tf.cast(dQdz_del, tf.complex64)
        bdQdz_del = tf.math.conj(dQdz_del)

        # ---- delete that column from ddQdz -> (B, ncoords, ncoords-1) ----
        ddQdz_del = tf.gather(ddQdz, gather_idx, batch_dims=1, axis=2)
        ddQdz_del = tf.cast(ddQdz_del, tf.complex64)
        bddQdz_del = tf.math.conj(ddQdz_del)

        # ---- delete that row+col from gfs -> (B, ncoords-1, ncoords-1) ----
        gfs_rows = tf.gather(gfs, gather_idx, batch_dims=1, axis=1)
        gfs_del  = tf.gather(gfs_rows, gather_idx, batch_dims=1, axis=2)
        gfs_del  = tf.cast(gfs_del, tf.complex64)

        g_inv = tf.linalg.inv(gfs_del)  # (B, ncoords-1, ncoords-1)

        # ---- H = bdQdz_del^T * g_inv * dQdz_del ----
        H = tf.einsum('bij,bi,bj->b', g_inv, bdQdz_del, dQdz_del)
        H_inv = tf.cast(1.0, H.dtype) / H

        # ---- qs ----
        qI = tf.expand_dims(qI, 0)
        qs = tf.math.pow(points, qI)
        qs = -5.0 * tf.reduce_prod(qs, axis=-1)  # (B,)

        # ---- theta ----
        theta = -tf.einsum('b,bij,bi,b->bj', H_inv, g_inv, bdQdz_del, qs)  # (B, ncoords-1)

        theta_real = tf.stack([tf.math.real(theta), tf.math.imag(theta)], axis=-1)  # (B, ncoords-1, 2)

    # jacobian wrt pts: (B, ncoords-1, 2, 2*ncoords) (typical)
    d_theta = tape.batch_jacobian(theta_real, pts, experimental_use_pfor=True)
    return d_theta


def calc_d_theta_par(target_file, pts, DQDZB, DQDZF, DDQDZB, DDQDZF, qI,
                     ncoords, comp_model, pg, batch_size=1000):
    # ---- FAST PATH: load NPZ cache ----
    if os.path.exists(target_file):
        data_theta = np.load(target_file)
        return tf.cast(data_theta["d_thetas"], dtype=tf.float32)

    pts = tf.convert_to_tensor(pts, dtype=tf.float32)
    n = int(pts.shape[0])
    nb = (n + batch_size - 1) // batch_size

    chunks = []
    for i in tqdm(range(nb)):
        b = pts[i * batch_size:(i + 1) * batch_size]
        chunks.append(calc_d_theta_batch(b, DQDZB, DQDZF, DDQDZB, DDQDZF, qI, ncoords, comp_model))

    d_thetas = tf.concat(chunks, axis=0)

    # Save in the same format you already use
    np.savez(target_file, d_thetas=d_thetas.numpy())

    return tf.cast(d_thetas, dtype=tf.float32)


class SpectralFSModel(PhiFSModel):
    def __init__(self, *args, **kwargs):
        deg_value = kwargs.pop("deg")
        self.monomials = kwargs.pop("monomials")

        ambient = kwargs.get("BASIS", {}).get("AMBIENT")
        ambient_len = len(ambient) if ambient is not None else 1
        if isinstance(deg_value, (list, tuple, np.ndarray)):
            deg_list = list(deg_value)
            self.k = deg_list if len(deg_list) != 1 or ambient_len == 1 else deg_list * ambient_len
        else:
            self.k = [deg_value for _ in range(ambient_len)]

        super().__init__(*args, **kwargs)

        self._generate_sections(self.k)
        self.learn_kaehler = tf.cast(False, dtype=tf.bool)
        self.learn_transition = tf.cast(False, dtype=tf.bool)
        self.learn_ricci = tf.cast(False, dtype=tf.bool)
        self.learn_ricci_val = tf.cast(False, dtype=tf.bool)
        self.learn_volk = tf.cast(False, dtype=tf.bool)

    def feature_engineered_call(self, pts):
        c_pts = tf.complex(pts[:, : pts.shape[-1] // 2], pts[:, pts.shape[-1] // 2 :])
        eig_funcs = self.get_eigenfunction_basis(c_pts)
        eig_funcs = tf.concat((tf.math.real(eig_funcs), tf.math.imag(eig_funcs)), axis=-1)
        return self.model(eig_funcs, training=False)

    def call(self, input_tensor, training=True, j_elim=None):
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(input_tensor)
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(input_tensor)
                phi = self.feature_engineered_call(input_tensor)
            d_phi = tape2.gradient(phi, input_tensor)
        dd_phi = tape1.batch_jacobian(d_phi, input_tensor)

        dx_dx_phi, dx_dy_phi, dy_dx_phi, dy_dy_phi = (
            0.25 * dd_phi[:, : self.ncoords, : self.ncoords],
            0.25 * dd_phi[:, : self.ncoords, self.ncoords :],
            0.25 * dd_phi[:, self.ncoords :, : self.ncoords],
            0.25 * dd_phi[:, self.ncoords :, self.ncoords :],
        )
        dd_phi = tf.complex(dx_dx_phi + dy_dy_phi, dx_dy_phi - dy_dx_phi)
        pbs = self.pullbacks(input_tensor, j_elim=j_elim)
        dd_phi = tf.einsum("xai,xij,xbj->xab", pbs, dd_phi, tf.math.conj(pbs))

        fs_cont = self.fubini_study_pb(input_tensor, pb=pbs, j_elim=j_elim)
        return tf.math.add(fs_cont, dd_phi)

    @staticmethod
    def get_num_sections(n, k):
        return math.comb(n + k - 1, k) if k < n else math.comb(n + k - 1, k) - math.comb(k - 1, k - n)

    @staticmethod
    def get_levicivita_tensor(dim):
        lc = np.zeros(tuple(dim for _ in range(dim)))
        for t in it.permutations(range(dim), r=dim):
            lc[t] = LeviCivita(*t)
        return tf.cast(lc, dtype=tf.complex128)

    def get_eigenfunction_basis(self, c_pts):
        c_pts = tf.cast(c_pts, tf.complex128)
        s_i = self.eval_sections_vec(c_pts)
        bs_j = self.eval_sections_vec(tf.math.conj(c_pts))
        sbs = tf.reshape(tf.einsum("xi,xj->xij", s_i, bs_j), (-1, s_i.shape[-1] ** 2))
        return tf.einsum("xa,x->xa", sbs, 1.0 / tf.einsum("xi,xi->x", c_pts, tf.math.conj(c_pts)) ** self.k)

    @staticmethod
    def _to_python_int(value, name="value"):
        """Convert scalar Tensor/NumPy values to a Python int for Python-side loops."""

        if isinstance(value, tf.Tensor):
            static_val = tf.get_static_value(value)
            if static_val is not None:
                return int(static_val)
            if tf.executing_eagerly():
                return int(value.numpy())
            raise TypeError(f"{name} must be statically known for section generation")
        return int(value)

    @classmethod
    def _to_int_list(cls, values, name="values"):
        """Convert scalar/vector Tensor/NumPy/list inputs to a plain Python int list."""

        if isinstance(values, tf.Tensor):
            static_vals = tf.get_static_value(values)
            if static_vals is None:
                if tf.executing_eagerly():
                    static_vals = values.numpy()
                else:
                    raise TypeError(f"{name} must be statically known for section generation")
            arr = np.asarray(static_vals).reshape(-1)
            return [int(v) for v in arr]
        return [cls._to_python_int(v, name=name) for v in list(values)]

    def generate_monomials(self, n, deg):
        n = self._to_python_int(n, name="n")
        deg = self._to_python_int(deg, name="deg")
        if n == 1:
            yield (deg,)
        else:
            for i in range(deg + 1):
                for j in self.generate_monomials(n - 1, deg - i):
                    yield (i,) + j

    def _generate_sections(self, k, ambient=False):
        self.sections = None
        k_list = self._to_int_list(k, name="k")
        degrees_list = self._to_int_list(self.degrees, name="degrees")

        ambient_polys = [0 for _ in range(len(k_list))]
        for i in range(len(k_list)):
            ambient_polys[i] = list(self.generate_monomials(degrees_list[i], k_list[i]))

        monomial_basis = [x for x in ambient_polys[0]]
        for i in range(1, len(k_list)):
            lenB = len(monomial_basis)
            monomial_basis = monomial_basis * len(ambient_polys[i])
            for l in range(len(ambient_polys[i])):
                for j in range(lenB):
                    monomial_basis[l * lenB + j] = monomial_basis[l * lenB + j] + ambient_polys[i][l]

        sections = np.array(monomial_basis, dtype=np.int64)
        if not ambient:
            reduced = np.unique(np.where(sections - self.monomials[0] < -0.1)[0])
            sections = sections[reduced]

        self.sections = tf.cast(sections, tf.complex128)
        self.nsections = len(self.sections)

    def eval_sections_vec(self, points):
        return tf.reduce_prod(tf.math.pow(tf.expand_dims(points, 1), self.sections), axis=-1)


class SpectralFSModelComp(SpectralFSModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, input_tensor, training=True, j_elim=None):
        return self.fubini_study_pb(input_tensor, j_elim=j_elim)

    def _fubini_study_n_metrics_tf(points, n=None, t=tf.complex(1.0, 0.0)):
        point_square = tf.reduce_sum(tf.abs(points) ** 2, axis=-1)
        outer = tf.einsum("xi,xj->xij", tf.math.conj(points), points)
        gFS = tf.einsum("x,ij->xij", point_square, tf.eye(points.shape[1]))
        outer = tf.cast(outer, dtype=tf.complex128)
        gFS = tf.cast(gFS, dtype=tf.complex128) - outer
        pt_squ = tf.cast(1 / (point_square**2), dtype=tf.complex128)
        vol_j = tf.cast(tf.math.real(t / np.pi), dtype=tf.complex128)
        return tf.einsum("xij,x->xij", gFS, pt_squ) * vol_j

    def fubini_study_amb(self, points, pb=None, j_elim=None, ts=None):
        del pb, j_elim
        if ts is None:
            ts = self.BASIS["KMODULI"]

        if self.nProjective > 1:
            cpoints = tf.complex(points[:, : self.degrees[0]], points[:, self.ncoords : self.ncoords + self.degrees[0]])
            fs = self._fubini_study_n_metrics(cpoints, n=self.degrees[0], t=ts[0])
            fs = tf.einsum("xij,ia,bj->xab", fs, self.proj_matrix["0"], tf.transpose(self.proj_matrix["0"]))
            for i in range(1, self.nProjective):
                s = tf.reduce_sum(self.degrees[:i])
                e = s + self.degrees[i]
                cpoints = tf.complex(points[:, s:e], points[:, self.ncoords + s : self.ncoords + e])
                fs_tmp = self._fubini_study_n_metrics(cpoints, n=self.degrees[i], t=ts[i])
                fs_tmp = tf.einsum(
                    "xij,ia,bj->xab",
                    fs_tmp,
                    self.proj_matrix[str(i)],
                    tf.transpose(self.proj_matrix[str(i)]),
                )
                fs += fs_tmp
        else:
            cpoints = tf.complex(points[:, : self.ncoords], points[:, self.ncoords : 2 * self.ncoords])
            fs = self._fubini_study_n_metrics(cpoints, t=ts[0])

        return fs


class BoostedPhiModel(PhiFSModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.models = [x for x in self.model]
        self.model = self.models[-1] if len(self.models) > 0 else None

    def call(self, input_tensor, training=True, j_elim=None):
        pbs = self.pullbacks(input_tensor, j_elim=j_elim)
        metr = self.fubini_study_pb(input_tensor, pb=pbs, j_elim=j_elim)
        for model in self.models:
            with tf.GradientTape(persistent=False) as tape1:
                tape1.watch(input_tensor)
                with tf.GradientTape(persistent=False) as tape2:
                    tape2.watch(input_tensor)
                    phi = model(input_tensor, training=False)
                d_phi = tape2.gradient(phi, input_tensor)
            dd_phi = tape1.batch_jacobian(d_phi, input_tensor)
            dx_dx_phi, dx_dy_phi, dy_dx_phi, dy_dy_phi = (
                0.25 * dd_phi[:, : self.ncoords, : self.ncoords],
                0.25 * dd_phi[:, : self.ncoords, self.ncoords :],
                0.25 * dd_phi[:, self.ncoords :, : self.ncoords],
                0.25 * dd_phi[:, self.ncoords :, self.ncoords :],
            )
            dd_phi = tf.complex(dx_dx_phi + dy_dy_phi, dx_dy_phi - dy_dx_phi)
            dd_phi = tf.einsum("xai,xij,xbj->xab", pbs, dd_phi, tf.math.conj(pbs))
            metr = tf.math.add(metr, dd_phi)
        return metr


class BoostedSpectralPhiModel(SpectralFSModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.models = [x for x in self.model]
        self.model = self.models[-1] if len(self.models) > 0 else None

    def feature_engineered_call(self, pts, model, training=False):
        c_pts = tf.complex(pts[:, : pts.shape[-1] // 2], pts[:, pts.shape[-1] // 2 :])
        eig_funcs = self.get_eigenfunction_basis(c_pts)
        eig_funcs = tf.concat((tf.math.real(eig_funcs), tf.math.imag(eig_funcs)), axis=-1)
        return model(eig_funcs, training=training)

    def call(self, input_tensor, training=True, j_elim=None):
        pbs = self.pullbacks(input_tensor, j_elim=j_elim)
        metr = self.fubini_study_pb(input_tensor, pb=pbs, j_elim=j_elim)
        for model in self.models:
            with tf.GradientTape(persistent=False) as tape1:
                tape1.watch(input_tensor)
                with tf.GradientTape(persistent=False) as tape2:
                    tape2.watch(input_tensor)
                    phi = self.feature_engineered_call(input_tensor, model, training=training)
                d_phi = tape2.gradient(phi, input_tensor)
            dd_phi = tape1.batch_jacobian(d_phi, input_tensor)
            dx_dx_phi, dx_dy_phi, dy_dx_phi, dy_dy_phi = (
                0.25 * dd_phi[:, : self.ncoords, : self.ncoords],
                0.25 * dd_phi[:, : self.ncoords, self.ncoords :],
                0.25 * dd_phi[:, self.ncoords :, : self.ncoords],
                0.25 * dd_phi[:, self.ncoords :, self.ncoords :],
            )
            dd_phi = tf.complex(dx_dx_phi + dy_dy_phi, dx_dy_phi - dy_dx_phi)
            dd_phi = tf.einsum("xai,xij,xbj->xab", pbs, dd_phi, tf.math.conj(pbs))
            metr = tf.math.add(metr, dd_phi)
        return metr

    def fubini_study_amb(self, points, pb=None, j_elim=None, ts=None):
        del pb, j_elim
        if ts is None:
            ts = self.BASIS["KMODULI"]

        if self.nProjective > 1:
            cpoints = tf.complex(points[:, : self.degrees[0]], points[:, self.ncoords : self.ncoords + self.degrees[0]])
            fs = self._fubini_study_n_metrics(cpoints, n=self.degrees[0], t=ts[0])
            fs = tf.einsum("xij,ia,bj->xab", fs, self.proj_matrix["0"], tf.transpose(self.proj_matrix["0"]))
            for i in range(1, self.nProjective):
                s = tf.reduce_sum(self.degrees[:i])
                e = s + self.degrees[i]
                cpoints = tf.complex(points[:, s:e], points[:, self.ncoords + s : self.ncoords + e])
                fs_tmp = self._fubini_study_n_metrics(cpoints, n=self.degrees[i], t=ts[i])
                fs_tmp = tf.einsum(
                    "xij,ia,bj->xab",
                    fs_tmp,
                    self.proj_matrix[str(i)],
                    tf.transpose(self.proj_matrix[str(i)]),
                )
                fs += fs_tmp
        else:
            cpoints = tf.complex(points[:, : self.ncoords], points[:, self.ncoords : 2 * self.ncoords])
            fs = self._fubini_study_n_metrics(cpoints, t=ts[0])

        return fs


class BoostedPhiModel_ambient(PhiFSModel):
    """Boosted Phi model that returns the ambient-space metric correction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.models = [x for x in self.model]
        self.model = self.models[-1] if len(self.models) > 0 else None

    def call(self, input_tensor, training=True, j_elim=None):
        metr = self.fubini_study_amb(input_tensor, pb=None, j_elim=j_elim)
        for model in self.models:
            with tf.GradientTape(persistent=False) as tape1:
                tape1.watch(input_tensor)
                with tf.GradientTape(persistent=False) as tape2:
                    tape2.watch(input_tensor)
                    phi = model(input_tensor, training=False)
                d_phi = tape2.gradient(phi, input_tensor)
            dd_phi = tape1.batch_jacobian(d_phi, input_tensor)
            dx_dx_phi, dx_dy_phi, dy_dx_phi, dy_dy_phi = (
                0.25 * dd_phi[:, : self.ncoords, :self.ncoords],
                0.25 * dd_phi[:, : self.ncoords, self.ncoords :],
                0.25 * dd_phi[:, self.ncoords :, : self.ncoords],
                0.25 * dd_phi[:, self.ncoords :, self.ncoords :],
            )
            dd_phi = tf.complex(dx_dx_phi + dy_dy_phi, dx_dy_phi - dy_dx_phi)
            metr = tf.math.add(metr, dd_phi)
        return metr

    def fubini_study_amb(self, points, pb=None, j_elim=None, ts=None):
        del pb, j_elim
        if ts is None:
            ts = self.BASIS["KMODULI"]

        if self.nProjective > 1:
            cpoints = tf.complex(points[:, : self.degrees[0]], points[:, self.ncoords : self.ncoords + self.degrees[0]])
            fs = self._fubini_study_n_metrics(cpoints, n=self.degrees[0], t=ts[0])
            fs = tf.einsum("xij,ia,bj->xab", fs, self.proj_matrix["0"], tf.transpose(self.proj_matrix["0"]))
            for i in range(1, self.nProjective):
                s = tf.reduce_sum(self.degrees[:i])
                e = s + self.degrees[i]
                cpoints = tf.complex(points[:, s:e], points[:, self.ncoords + s : self.ncoords + e])
                fs_tmp = self._fubini_study_n_metrics(cpoints, n=self.degrees[i], t=ts[i])
                fs_tmp = tf.einsum(
                    "xij,ia,bj->xab",
                    fs_tmp,
                    self.proj_matrix[str(i)],
                    tf.transpose(self.proj_matrix[str(i)]),
                )
                fs += fs_tmp
        else:
            cpoints = tf.complex(points[:, : self.ncoords], points[:, self.ncoords : 2 * self.ncoords])
            fs = self._fubini_study_n_metrics(cpoints, t=ts[0])

        return fs


class BoostedSpectralPhiModel_ambient(SpectralFSModel):
    """Boosted spectral Phi model that returns the ambient-space metric correction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.models = [x for x in self.model]
        self.model = self.models[-1] if len(self.models) > 0 else None

    def feature_engineered_call(self, pts, model, training=False):
        c_pts = tf.complex(pts[:, : pts.shape[-1] // 2], pts[:, pts.shape[-1] // 2 :])
        eig_funcs = self.get_eigenfunction_basis(c_pts)
        eig_funcs = tf.concat((tf.math.real(eig_funcs), tf.math.imag(eig_funcs)), axis=-1)
        return model(eig_funcs, training=training)

    def call(self, input_tensor, training=True, j_elim=None):
        metr = self.fubini_study_amb(input_tensor, pb=None, j_elim=j_elim)
        for model in self.models:
            with tf.GradientTape(persistent=False) as tape1:
                tape1.watch(input_tensor)
                with tf.GradientTape(persistent=False) as tape2:
                    tape2.watch(input_tensor)
                    phi = self.feature_engineered_call(input_tensor, model, training=training)
                d_phi = tape2.gradient(phi, input_tensor)
            dd_phi = tape1.batch_jacobian(d_phi, input_tensor)
            dx_dx_phi, dx_dy_phi, dy_dx_phi, dy_dy_phi = (
                0.25 * dd_phi[:, : self.ncoords, :self.ncoords],
                0.25 * dd_phi[:, : self.ncoords, self.ncoords :],
                0.25 * dd_phi[:, self.ncoords :, : self.ncoords],
                0.25 * dd_phi[:, self.ncoords :, self.ncoords :],
            )
            dd_phi = tf.complex(dx_dx_phi + dy_dy_phi, dx_dy_phi - dy_dx_phi)
            metr = tf.math.add(metr, dd_phi)
        return metr

    def fubini_study_amb(self, points, pb=None, j_elim=None, ts=None):
        del pb, j_elim
        if ts is None:
            ts = self.BASIS["KMODULI"]

        if self.nProjective > 1:
            cpoints = tf.complex(points[:, : self.degrees[0]], points[:, self.ncoords : self.ncoords + self.degrees[0]])
            fs = self._fubini_study_n_metrics(cpoints, n=self.degrees[0], t=ts[0])
            fs = tf.einsum("xij,ia,bj->xab", fs, self.proj_matrix["0"], tf.transpose(self.proj_matrix["0"]))
            for i in range(1, self.nProjective):
                s = tf.reduce_sum(self.degrees[:i])
                e = s + self.degrees[i]
                cpoints = tf.complex(points[:, s:e], points[:, self.ncoords + s : self.ncoords + e])
                fs_tmp = self._fubini_study_n_metrics(cpoints, n=self.degrees[i], t=ts[i])
                fs_tmp = tf.einsum(
                    "xij,ia,bj->xab",
                    fs_tmp,
                    self.proj_matrix[str(i)],
                    tf.transpose(self.proj_matrix[str(i)]),
                )
                fs += fs_tmp
        else:
            cpoints = tf.complex(points[:, : self.ncoords], points[:, self.ncoords : 2 * self.ncoords])
            fs = self._fubini_study_n_metrics(cpoints, t=ts[0])

        return fs


class HarmonicFormModelCachedPB(tfk.Model):
    """
    Training model that uses precomputed (pulled-back) geometry and per-sample pullbacks pbs.
    Only NN derivatives are computed online.

    Batch inputs:
      x: (B, 2*ncoords) float32

    Batch targets:
      y: complex rhs tensor (same shape as lhs_symm)

    Batch cache (all aligned with x rows, already pulled back):
      pbs:          (B, n_fold, ncoords) complex64
      g_cy_pb:      (B, n_fold, n_fold) complex64
      g_inv_pb:     (B, n_fold, n_fold) complex64
      d_gcy_z_pb:   (B, ncoords, n_fold, n_fold) complex64
      d_ginv_z_pb:  (B, ncoords, n_fold, n_fold) complex64
      omega_pb:     (B,) complex64     (or (B, ...) if yours is structured)
      d_omega_z_pb: (B, ncoords) complex64 (or (B, ncoords, ...) if yours is structured)
    """

    def __init__(self, nn, *, ncoords, n_fold, levi, clip_norm=1.0):
        super().__init__()
        self.nn = nn
        self.ncoords = int(ncoords)
        self.n_fold = int(n_fold)
        self.levi = tf.cast(levi, tf.complex64)
        self.clip_norm = float(clip_norm)

    def call(self, x, training=False):
        return self.nn(x, training=training)

    @tf.function(jit_compile=False)
    def get_lhs_cached_pb(
        self,
        x,
        *,
        pbs,
        g_cy_pb,
        g_inv_pb,
        d_gcy_z_pb,
        d_ginv_z_pb,
        omega_pb,
        d_omega_z_pb,
        training=True,
    ):
        """
        Compute lhs_symm using:
          - NN output derivatives (computed inside tapes)
          - cached pulled-back geometry
          - pbs to pull back NN-derivatives
        """
        ncoords = self.ncoords
        n_fold = self.n_fold
        levi = self.levi

        # Cast caches (no gradients needed through these)
        pbs          = tf.cast(pbs, tf.complex64)
        bpbs         = tf.math.conj(pbs)

        g_cy_pb      = tf.cast(g_cy_pb, tf.complex64)
        g_inv_pb     = tf.cast(g_inv_pb, tf.complex64)
        d_gcy_z_pb   = tf.cast(d_gcy_z_pb, tf.complex64)
        d_ginv_z_pb  = tf.cast(d_ginv_z_pb, tf.complex64)
        omega_pb     = tf.cast(omega_pb, tf.complex64)
        d_omega_z_pb = tf.cast(d_omega_z_pb, tf.complex64)

        # -------- NN derivatives (correct tape usage) --------
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x)
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                y_pred = self.nn(x, training=training)
            grads = tape2.batch_jacobian(y_pred, x)

        jacs = tape1.batch_jacobian(grads, x)
        del tape1


        # Your NN outputs n_out = 2*n_fold: first half real, second half imag
        grads = tf.cast(grads, tf.float32)
        jacs  = tf.cast(jacs, tf.complex64)

        # -------- build complex \bar∂ y_pred / \bar∂ z --------
        re_dz_re = grads[:, 0:n_fold, :ncoords]
        im_dz_re = grads[:, n_fold:, :ncoords]
        re_dz_im = grads[:, 0:n_fold, ncoords:]
        im_dz_im = grads[:, n_fold:, ncoords:]
        bd_y_pred_bz = 0.5 * tf.complex(re_dz_re - im_dz_im, im_dz_re + re_dz_im)  # (B, n_fold, ncoords)

        # -------- build complex (∂)(\bar∂) y_pred --------
        dd_y_pred = jacs[:, 0:n_fold, 0:ncoords, 0:ncoords]
        dd_y_pred += 1.0j * jacs[:, n_fold:, 0:ncoords, 0:ncoords]
        dd_y_pred += 1.0j * jacs[:, 0:n_fold, ncoords:, 0:ncoords]
        dd_y_pred -=        jacs[:, n_fold:, ncoords:, 0:ncoords]
        dd_y_pred -= 1.0j * jacs[:, 0:n_fold, 0:ncoords, ncoords:]
        dd_y_pred +=        jacs[:, n_fold:, 0:ncoords, ncoords:]
        dd_y_pred +=        jacs[:, 0:n_fold, ncoords:, ncoords:]
        dd_y_pred += 1.0j * jacs[:, n_fold:, ncoords:, ncoords:]
        dd_y_pred *= 0.25  # (B, n_fold, ncoords, ncoords)

        # -------- pull back NN derivatives using pbs/bpbs --------
        # same as your original:
        # dd_y_pred_pb = einsum('xai,xbj,xkji->xkba', pbs, bpbs, dd_y_pred)
        dd_y_pred_pb = tf.einsum('xai,xbj,xkji->xkba', pbs, bpbs, dd_y_pred)

        # bd_y_pred_bz_pb = einsum('xai,xbi->xab', bpbs, bd_y_pred_bz)
        bd_y_pred_bz_pb = tf.einsum('xai,xbi->xab', bpbs, bd_y_pred_bz)

        # -------- contractions (all in PB frame now) --------
        lhs_xi  = tf.einsum('x,abm,xlm,xsk,xsln->xabnk', omega_pb, levi, g_inv_pb, g_cy_pb, dd_y_pred_pb)
        lhs_xi += tf.einsum('x,abm,xlm,xnsk,xls->xabnk', omega_pb, levi, g_inv_pb, d_gcy_z_pb, bd_y_pred_bz_pb)
        lhs_xi += tf.einsum('x,abm,xnlm,xsk,xls->xabnk', omega_pb, levi, d_ginv_z_pb, g_cy_pb, bd_y_pred_bz_pb)
        lhs_xi += tf.einsum('xn,abm,xlm,xsk,xls->xabnk', d_omega_z_pb, levi, g_inv_pb, g_cy_pb, bd_y_pred_bz_pb)

        lhs_symm = (lhs_xi
                    + tf.transpose(lhs_xi, perm=[0, 2, 3, 1, 4])
                    + tf.transpose(lhs_xi, perm=[0, 3, 1, 2, 4])) / 3.0
        return lhs_symm

    def train_step(self, data):
        """
        data:
          x, (y, cache_dict)
        """
        x, (y, cache) = data

        # unpack cache (dict expected)
        pbs          = cache["pbs"]
        g_cy_pb      = cache["g_cy_pb"]
        g_inv_pb     = cache["g_inv_pb"]
        d_gcy_z_pb   = cache["d_gcy_z_pb"]
        d_ginv_z_pb  = cache["d_ginv_z_pb"]
        omega_pb     = cache["omega_pb"]
        d_omega_z_pb = cache["d_omega_z_pb"]

        with tf.GradientTape() as tape:
            lhs = self.get_lhs_cached_pb(
                x,
                pbs=pbs,
                g_cy_pb=g_cy_pb,
                g_inv_pb=g_inv_pb,
                d_gcy_z_pb=d_gcy_z_pb,
                d_ginv_z_pb=d_ginv_z_pb,
                omega_pb=omega_pb,
                d_omega_z_pb=d_omega_z_pb,
                training=True,
            )

            # match your original loss convention
            lhs_real = tf.stack([tf.math.real(lhs), tf.math.imag(lhs)], axis=-1)
            y_real   = tf.stack([tf.math.real(y),   tf.math.imag(y)],   axis=-1)
            loss = tf.reduce_mean(tf.abs(lhs_real + y_real))

        train_vars = self.nn.trainable_variables
        grads = tape.gradient(loss, train_vars)

        # sanitize grads
        adjusted_grads = []
        for g, v in zip(grads, train_vars):
            if g is None:
                raise ValueError(f"Gradient for {v.name} is None (check graph connectivity).")
            g = tf.where(tf.math.is_nan(g), tf.zeros_like(g), g)
            if self.clip_norm and self.clip_norm > 0:
                g = tf.clip_by_norm(g, self.clip_norm)
            adjusted_grads.append(g)

        self.optimizer.apply_gradients(zip(adjusted_grads, train_vars))
        self.compiled_metrics.update_state(y, -lhs)
        return {"loss": loss, **{m.name: m.result() for m in self.metrics}}


class WarpFactorModel(tfk.Model):
    def __init__(self, tfmodel, g_cy_model):
        super(WarpFactorModel, self).__init__()
        self.model = tfmodel
        self.g_cy_model = g_cy_model
    
    def call(self, input_tensor):
        return tf.math.exp(-4. * self.model(input_tensor))

    def compile(self, **kwargs):
        super(WarpFactorModel, self).compile(**kwargs)

    def train_step(self, data):
        x, y = data

        with tf.GradientTape(persistent=False) as tape:
            trainable_vars = self.model.trainable_variables
            tape.watch(trainable_vars)
            laplace_op = self.get_lhs(x, y)
            loss = tf.reduce_mean(tf.math.abs(y - laplace_op))
        
        # Compute gradients
        gradients = tape.gradient(loss, trainable_vars)
        # remove nans and gradient clipping from transition loss.
        gradients = [tf.where(tf.math.is_nan(g), 1e-8, g) for g in gradients]
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        self.compiled_metrics.update_state(y, laplace_op)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def get_lhs(self, x, y):
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x)
            with tf.GradientTape(persistent=False) as tape2:
                tape2.watch(x)
                y_pred = self(x)
            grads = tape2.gradient(y_pred, x)
        jacs = tf.cast(tape1.batch_jacobian(grads, x), dtype=tf.complex64)
        # add derivatives together to complex tensor
        delta_ij = jacs[:, 0:self.g_cy_model.ncoords, 0:self.g_cy_model.ncoords]
        delta_ij += 1j*jacs[:, 0:self.g_cy_model.ncoords, self.g_cy_model.ncoords:]
        delta_ij -= 1j*jacs[:, self.g_cy_model.ncoords:, 0:self.g_cy_model.ncoords]
        delta_ij += jacs[:, self.g_cy_model.ncoords:, self.g_cy_model.ncoords:]
        delta_ij *= 0.25
        #tf.print(delta_ij)
        pullbacks = self.g_cy_model.pullbacks(x)
        g_inv = tf.linalg.inv(self.g_cy_model(x))
        laplace_op = tf.einsum('xba,xai,xij,xbj->x', -2 * g_inv, pullbacks, delta_ij, tf.math.conj(pullbacks))
        laplace_op = tf.math.real(laplace_op)
        #tf.print(laplace_op)
        
        return laplace_op


__all__ = [
    "BoostedPhiModel",
    "BoostedPhiModel_ambient",
    "BoostedSpectralPhiModel",
    "BoostedSpectralPhiModel_ambient",
    "ComplexFunctionConsumer",
    "KaehlerCallback",
    "PointGeneratorMathematicaIPS",
    "PointGeneratorMathematica",
    "PhiFSModel",
    "RicciCallback",
    "SigmaCallback",
    "SigmaLoss",
    "SpectralFSModel",
    "SpectralFSModelComp",
    "VolkCallback",
    "build_phi_network",
    "calculate_dQdz",
    "calculate_ddQdz",
    "calc_d_theta_par",
    "calc_d_theta_par_ambient",
    "calc_d_theta_batch_ambient",
    "christoffel",
    "levi_civita_tensor",
    "riemann",
    "compute_riemann",
    "compute_riemann_par_oomsafe",
    "get_chern_classes",
    "integrate_standard",
    "integrate_weighted",
    "compute_weil_petersson",
    "configure_tensorflow_runtime",
    "holomorphic_volume_form",
    "prepare_tf_basis",
    "pullbacks",
    "trace_after_row_insertion",
    "train_model",
    "calc_derivatives_fast",
    "make_dataset_cached_pb",
    "build_harm_nn",
    "HarmonicFormModelCachedPB",
    "trace_after_row_insertion",
    "calc_d_theta_batch",
    "calc_d_theta_par",
    "WarpFactorModel"
]
