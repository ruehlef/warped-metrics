(* ::Package:: *)

getFS[coords_,bcoords_,sections_:{},bsections_:{},h_:{},kval_:1]:=Module[{hh,s,bs,kk},(
hh=h;
s=sections;
bs=bsections;
kk=kval;
If[s=={},s=coords];
If[bs=={},bs=bcoords];
If[hh=={},hh=IdentityMatrix[Length[s]]];
If[!IntegerQ[kk],kk=1];
Return[1/(\[Pi] kk) Table[D[D[Log[bs . (hh . s)],bcoords[[j]]],coords[[i]]],{i,Length[coords]},{j,Length[bcoords]}]];
)];

getAbsMaxPos[alist_]:=Module[{k,maxPos},(
maxPos=1;
For[k=1,k<=Length[alist],k++,
If[Abs[alist[[k]]]>Abs[alist[[maxPos]]],maxPos=k];
];
Return[maxPos];
)];

SamplePoint[p_,L_,vars_,dimX_]:=Module[{pVars,aCoeffs,RandSections,randPatch,sols,xsols,i},(
RandSections={};
For[i=1,i<=dimX,i++,
aCoeffs=L . RandomVariate[NormalDistribution[0,1],Length[vars]]+I L . RandomVariate[NormalDistribution[0,1],Length[vars]];
AppendTo[RandSections,(Plus@@(vars aCoeffs))==0];
];
randPatch=RandomInteger[{1,Length[vars]}];
sols=Quiet[Solve[Join[{p==0},RandSections]/.{vars[[randPatch]]->1}]];
xsols=vars/.sols/.{vars[[randPatch]]->1};
xsols=Table[xsols[[i]]/xsols[[i,getAbsMaxPos[xsols[[i]]]]],{i,Length[xsols]}];
Return[xsols];
)];

getWeightOmegas[coords_,bcoords_,g_,p_,bp_,pts_,bpts_,patchIndex_,\[Kappa]_:1]:=Module[{\[Omega],\[Omega]PB,w,bw,dw,dbw,max,maxPos,maxPosGlobal,derivs,bderivs,localCoords,localbCoords,i,goodCoordsIndexSet,\[Omega]Top,OmegaOmegaBar,substCoords,substbCoords},(
localCoords=Delete[coords,patchIndex];
localbCoords=Delete[bcoords,patchIndex];
substCoords=Table[coords[[k]]->pts[[k]]/pts[[patchIndex]],{k,Length[pts]}];
substbCoords=Table[bcoords[[k]]->bpts[[k]]/bpts[[patchIndex]],{k,Length[bpts]}];

\[Omega]= g/.substCoords/.substbCoords;
(*Find max |d p/Subscript[dz, j]|*)
derivs=Table[D[p,localCoords[[i]]],{i,Length[localCoords]}]/.substCoords;
bderivs=Table[D[bp,localbCoords[[i]]],{i,Length[localbCoords]}]/.substbCoords;
maxPos=getAbsMaxPos[derivs];
maxPosGlobal=Position[coords,localCoords[[maxPos]]][[1,1]];
(*compute the Jacobians. These are d x n+1 matrices for a CY d-fold inside P^n *)
dw[a_,i_]:=
If[a==patchIndex,0,
If[a==maxPosGlobal,
-D[p,coords[[i]]]/derivs[[maxPos]]/.substCoords,
KroneckerDelta[a,i]
]
];
dbw[a_,i_]:=
If[a==patchIndex,0,
If[a==maxPosGlobal,-D[bp,bcoords[[i]]]/bderivs[[maxPos]]/.substbCoords,
KroneckerDelta[a,i]
]
];
goodCoordsIndexSet=DeleteCases[DeleteCases[Table[i,{i,Length[\[Omega]]}],maxPosGlobal],patchIndex];
\[Omega]PB=Table[Sum[dw[a,i] \[Omega][[a,b]] dbw[b,j],{a,Length[\[Omega]]},{b,Length[\[Omega]]}],{i,goodCoordsIndexSet},{j,goodCoordsIndexSet}];
\[Omega]Top=Length[\[Omega]PB]! Det[\[Omega]PB];
(*Now Omega wedge bOmega*)
OmegaOmegaBar=1/(derivs[[maxPos]] bderivs[[maxPos]]);
(*Now compute the weights*)
w=OmegaOmegaBar/\[Omega]Top;
Return[Abs[Chop[{\[Kappa] w,OmegaOmegaBar}]]];(*Should be real, but sometimes has a small imaginary part of the order of 10^-10 or so, which is not chopped*)
\.00)];

getFSPB[coords_,bcoords_,p_,bp_,pts_,bpts_,patchIndex_,kval_:1]:=Module[{\[Omega],\[Omega]PB,w,bw,dw,dbw,max,maxPos,maxPosGlobal,derivs,bderivs,localCoords,localbCoords,i,goodCoordsIndexSet,\[Omega]Top,OmegaOmegaBar,substCoords,substbCoords,g},(
localCoords=Delete[coords,patchIndex];
localbCoords=Delete[bcoords,patchIndex];
substCoords=Table[coords[[k]]->pts[[k]]/pts[[patchIndex]],{k,Length[pts]}];
substbCoords=Table[bcoords[[k]]->bpts[[k]]/bpts[[patchIndex]],{k,Length[bpts]}];
g=getFS[coords,bcoords,{},{},{},kval];
\[Omega]= g/.substCoords/.substbCoords;
(*Find max |d p/Subscript[dz, j]|*)
derivs=Table[D[p,localCoords[[i]]],{i,Length[localCoords]}]/.substCoords;
bderivs=Table[D[bp,localbCoords[[i]]],{i,Length[localbCoords]}]/.substbCoords;
maxPos=getAbsMaxPos[derivs];
maxPosGlobal=Position[coords,localCoords[[maxPos]]][[1,1]];
(*compute the Jacobians. These are d x n+1 matrices for a CY d-fold inside P^n *)
dw[a_,i_]:=
If[a==patchIndex,0,
If[a==maxPosGlobal,
-D[p,coords[[i]]]/derivs[[maxPos]]/.substCoords,
KroneckerDelta[a,i]
]
];
dbw[a_,i_]:=
If[a==patchIndex,0,
If[a==maxPosGlobal,-D[bp,bcoords[[i]]]/bderivs[[maxPos]]/.substbCoords,
KroneckerDelta[a,i]
]
];
goodCoordsIndexSet=DeleteCases[DeleteCases[Table[i,{i,Length[\[Omega]]}],maxPosGlobal],patchIndex];
\[Omega]PB=Table[Sum[dw[a,i] \[Omega][[a,b]] dbw[b,j],{a,Length[\[Omega]]},{b,Length[\[Omega]]}],{i,goodCoordsIndexSet},{j,goodCoordsIndexSet}];
Return[\[Omega]PB];
)];

GetNewLambdas[coords_,bcoords_,p_,bp_,pts_,Ls_,\[Kappa]s_,dimX_:1]:=Module[{gFS,allWeights,minData,maxData,minPoint,maxPoint,wMin,wMax,\[Epsilon]Min,\[Epsilon]Max,Px,\[Lambda]Min,\[Lambda]Max,\[Lambda]MinInv,\[Lambda]MaxInv,\[Kappa]Ref},(*Precompute standard FS metric once*)
gFS=getFS[coords,bcoords];
(*Compute all weights in parallel*)
allWeights=ParallelTable[
Module[{pt,patchIndex,wFS,\[CapitalOmega]FS,allMetricWeights},
pt=pts[[i,1]];
patchIndex=pts[[i,4]];
{wFS,\[CapitalOmega]FS}=getWeightOmegas[coords,bcoords,gFS,p,bp,pt,Conjugate[pt],patchIndex,\[Kappa]s[[1]]];
(*Calculate weights for all metrics*)
allMetricWeights=Table[With[{g=getFS[coords,bcoords,{},{},ConjugateTranspose[Ls[[j]]] . Ls[[j]]]},First[getWeightOmegas[coords,bcoords,g,p,bp,pt,Conjugate[pt],patchIndex,\[Kappa]Ref]]],{j,Length[Ls]}];
(*For each point,return: minimum weight, maximum weight, point, epsilon, index*)
{Min[allMetricWeights],Max[allMetricWeights],pt,wFS^(1/dimX)-1,i}],{i,Length[pts]}];
(*Find global minimum and maximum*)
minData=SortBy[allWeights,#[[1]]&][[1]];
maxData=SortBy[allWeights,-#[[2]]&][[1]];

(*Extract minimum and maximum data*)
wMin=minData[[1]];
minPoint=minData[[3]];
\[Epsilon]Min=minData[[4]];
wMax=maxData[[2]];
maxPoint=maxData[[3]];
\[Epsilon]Max=maxData[[4]];
(*Construct Lambdas*)
Px=KroneckerProduct[minPoint,ConjugateTranspose[minPoint]]/(ConjugateTranspose[minPoint] . minPoint);
\[Lambda]Min=1/(1+\[Epsilon]Min) (IdentityMatrix[Length[coords]]+\[Epsilon]Min Px);
Px=KroneckerProduct[maxPoint,ConjugateTranspose[maxPoint]]/(ConjugateTranspose[maxPoint] . maxPoint);
\[Lambda]Max=1/(1+\[Epsilon]Max) (IdentityMatrix[Length[coords]]+\[Epsilon]Max Px);
(*Hermitianize the inverse matrices*)
\[Lambda]MinInv=Chop[Inverse[Chop[\[Lambda]Min]]];
\[Lambda]MaxInv=Chop[Inverse[Chop[\[Lambda]Max]]];
\[Lambda]MinInv=Chop[1/2 (\[Lambda]MinInv+ConjugateTranspose[\[Lambda]MinInv])];
\[Lambda]MaxInv=Chop[1/2 (\[Lambda]MaxInv+ConjugateTranspose[\[Lambda]MaxInv])];
(*Print["Found 2 new \[Lambda]s"];
Print["With these \[Lambda]s, the new weights for the points used to construct them are:"];
Print["\[Lambda]Min:",getWeightOmegas[coords,bcoords,getFS[coords,bcoords],p,bp,minPoint,Conjugate[minPoint],getAbsMaxPos[minPoint],\[Kappa]s[[1]]],getWeightOmegas[coords,bcoords,getFS[coords,bcoords,{},{},\[Lambda]MinInv],p,bp,minPoint,Conjugate[minPoint],getAbsMaxPos[minPoint],\[Kappa]s[[1]]]];
Print["\[Lambda]Max:",getWeightOmegas[coords,bcoords,getFS[coords,bcoords],p,bp,maxPoint,Conjugate[maxPoint],getAbsMaxPos[maxPoint],\[Kappa]s[[1]]],getWeightOmegas[coords,bcoords,getFS[coords,bcoords,{},{},\[Lambda]MaxInv],p,bp,maxPoint,Conjugate[maxPoint],getAbsMaxPos[maxPoint],\[Kappa]s[[1]]]];*)

Return[{CholeskyDecomposition[\[Lambda]MinInv],CholeskyDecomposition[\[Lambda]MaxInv]}];
];

SamplePoints[poly_,bpoly_,L_,coords_,bcoords_,numPts_,dimX_,\[Kappa]In_:1]:=Module[{i,j,ptsNew,g,pt,patchIndex,w,\[CapitalOmega],res,pts,\[Kappa]},(
\[Kappa]=\[Kappa]In;
pts=ParallelTable[
ptsNew=SamplePoint[poly,L,coords,dimX];
g=getFS[coords,bcoords,{},{},ConjugateTranspose[L] . L];
res={};
For[j=1,j<=Length[ptsNew],j++,
pt=ptsNew[[j]];
patchIndex=getAbsMaxPos[pt];
{w,\[CapitalOmega]}=getWeightOmegas[coords,bcoords,g,poly,bpoly,pt,Conjugate[pt],patchIndex,\[Kappa]];
AppendTo[res,Chop[{pt,w,\[CapitalOmega],patchIndex,L}]]
];
res
,
{i,Ceiling[numPts/5]}
];
(*If no \[Kappa] was provided, estimate it based on the points we sampled and update the weights*)
pts=Flatten[pts,1];
If[\[Kappa]==1,
\[Kappa]=1/Mean[pts[[;;,2]]];
pts[[;;,2]]=pts[[;;,2]]*\[Kappa];
];
Return[{pts,\[Kappa]}];(*{point, w, \[CapitalOmega], patchIndex, L}*)
)];

SamplePointsWithRejection[poly_,bpoly_,Ls_,LPos_,coords_,bcoords_,numPts_, dimX_,startPts_:{},\[Kappa]In_:1]:=Module[{newPts,gs,allWeights,j,pts,numToSample,resampleCounter,\[Kappa]},(
resampleCounter=0;
newPts={};
(*Pre-compute the FS metrics*)
gs=Table[getFS[coords,bcoords,{},{},ConjugateTranspose[Ls[[j]]] . Ls[[j]]],{j,Length[Ls]}];
(* Get estimate for \[Kappa] in that region if none is provided *)
If[\[Kappa]In!=1,
\[Kappa]=\[Kappa]In;
ClientLibrary`info["Using \[Kappa]="<>ToString[\[Kappa]]];
,
\[Kappa]=SamplePoints[poly,bpoly,Ls[[LPos]],coords,bcoords,50000,dimX][[2]];
ClientLibrary`info["Calculated \[Kappa]="<>ToString[\[Kappa]]];
];

While[Length[newPts]<numPts,
(*If newPts != {}, check for and reject unfitting points that are already in there. Else, generate some*)
If[startPts!={}&&resampleCounter==0,
pts=startPts;
,
(*Sample Len[Ls] times more points than needed, since we need to throw away many. Keep number of sampled points between 5k and 20k *)
numToSample=Min[20000,Max[5000, Length[Ls] (numPts-Length[newPts])]];
pts=SamplePoints[poly,bpoly,Ls[[LPos]],coords,bcoords,numToSample,dimX][[1]];
];
(*Compute all weights for all points*)
allWeights={};
For[j=1,j<=Length[gs],j++,
AppendTo[allWeights,ParallelTable[getWeightOmegas[coords,bcoords,gs[[j]],poly,bpoly,pts[[i,1]],Conjugate[pts[[i,1]]],pts[[i,4]],\[Kappa]][[1]],{i,Length[pts]}]];
];
allWeights=Abs[Transpose[allWeights]-1];(*Now each row contains all Abs[\[Kappa] weights-1] for a given point*)
allWeights=ParallelTable[MemberQ[Flatten[Position[allWeights[[i]],Min[allWeights[[i]]]]],LPos],{i,Length[allWeights]}];
pts=Pick[pts,allWeights];
newPts=Join[newPts,pts];
If[startPts!={}&&resampleCounter==0,
ClientLibrary`info["Could reuse "<>ToString[Length[newPts]] <> "/" <> ToString[numPts] <> " points for metric " <> ToString[LPos] <> " / " <> ToString[Length[Ls]]];
,
ClientLibrary`info["Found "<>ToString[Length[newPts]] <> "/" <> ToString[numPts] <> " points (" <> ToString[Length[pts]] <> " new) on sample iteration " <> ToString[resampleCounter+1] <> " for metric " <> ToString[LPos] <> " / " <> ToString[Length[Ls]]];
];
resampleCounter+=1;
];
ClientLibrary`info["For metric " <> ToString[LPos] <> " had to resample " <> If[startPts!={},ToString[resampleCounter-1],ToString[resampleCounter]] <> " times"];

Return[{newPts[[;;numPts]],\[Kappa]}];
)];

GeneratePointsMIPS[TotalNumPts_,NumRegions_,\[Psi]_,dimX_,RejectionSampling_:False]:=Module[{NumPts,coords,bcoords,poly,bpoly,Ls,allPts,\[Kappa],\[Kappa]s,newPts,r,\[Psi]\[Psi],rescaleFactor,i,auxWeights,normFac,DetFS},(
ClientLibrary`SetInfoLogLevel[];
NumPts=Ceiling[TotalNumPts/NumRegions];
ClientLibrary`info["Generating " <> ToString[NumPts] <> " points in each of the " <> ToString[NumRegions] <> " regions for a total of " <> ToString[TotalNumPts] <>" points."];
coords={Subscript[z, 0],Subscript[z, 1],Subscript[z, 2],Subscript[z, 3],Subscript[z, 4]};
bcoords={Subscript[bz, 0],Subscript[bz, 1],Subscript[bz, 2],Subscript[bz, 3],Subscript[bz, 4]};
poly=Subscript[z, 0]^5+Subscript[z, 1]^5+Subscript[z, 2]^5+Subscript[z, 3]^5+Subscript[z, 4]^5-5\[Psi]\[Psi] Subscript[z, 0] Subscript[z, 1] Subscript[z, 2] Subscript[z, 3] Subscript[z, 4]/.{\[Psi]\[Psi]->\[Psi]};
bpoly=Subscript[bz, 0]^5+Subscript[bz, 1]^5+Subscript[bz, 2]^5+Subscript[bz, 3]^5+Subscript[bz, 4]^5-5Conjugate[\[Psi]\[Psi]] Subscript[bz, 0] Subscript[bz, 1] Subscript[bz, 2] Subscript[bz, 3] Subscript[bz, 4]/.{\[Psi]\[Psi]->\[Psi]};
Ls={IdentityMatrix[Length[coords]]};
ClientLibrary`info["Processing region " <> ToString[1]];
{allPts,\[Kappa]}=SamplePoints[poly,bpoly,Ls[[-1]],coords,bcoords,NumPts,dimX];
\[Kappa]s={\[Kappa]};
ClientLibrary`info["Calculated \[Kappa]="<>ToString[\[Kappa]]];
For[r=1,r<=Floor[(NumRegions-1)/2],r++,
Ls=Join[Ls,GetNewLambdas[coords,bcoords,poly,bpoly,allPts,Ls,\[Kappa]s,dimX]];
ClientLibrary`info["Processing region " <> ToString[2*r]];
If[RejectionSampling,
{newPts,\[Kappa]}=SamplePointsWithRejection[poly,bpoly,Ls,Length[Ls]-1,coords,bcoords,NumPts,dimX];,
{newPts,\[Kappa]}=SamplePoints[poly,bpoly,Ls[[-2]],coords,bcoords,NumPts,dimX];
];
allPts=Join[allPts,newPts];
AppendTo[\[Kappa]s,\[Kappa]];
ClientLibrary`info["Processing region " <> ToString[2*r+1]];
If[RejectionSampling,
{newPts,\[Kappa]}=SamplePointsWithRejection[poly,bpoly,Ls,Length[Ls],coords,bcoords,NumPts,dimX];,
{newPts,\[Kappa]}=SamplePoints[poly,bpoly,Ls[[-1]],coords,bcoords,NumPts,dimX];
];
allPts=Join[allPts,newPts];
AppendTo[\[Kappa]s,\[Kappa]];
];
(* DO this in Python
ClientLibrary`info["Normalizing weights..."];
(* Compute determinants with unit kappa for proper normalization *)
DetFS=ParallelTable[Abs[Det[getFSPB[coords,bcoords,poly,bpoly,allPts[[i,1]],Conjugate[allPts[[i,1]]],getAbsMaxPos[allPts[[i,1]]],1]]],{i,Length[allPts]}];
(* The factor 5 comes from the degree of the Calabi-Yau (quintic) *)
(* Normalize to get proper measure weights
normalizationFactor=5/Mean[DetFS allPts[[;;,2]]];
allPts[[;;,2]]=allPts[[;;,2]]*normalizationFactor;
*)
*)
ClientLibrary`info["done"];
Return[{allPts[[;;,1]],allPts[[;;,2]],allPts[[;;,3]],\[Kappa]s,{3}}];
)];
