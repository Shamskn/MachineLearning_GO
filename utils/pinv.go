package utils

import (
	"gonum.org/v1/gonum/mat"
)

func Pinv(a *mat.Dense) *mat.Dense {
	var svd mat.SVD
	svd.Factorize(a, mat.SVDThin)
	V := svd.VTo(nil)
	U := svd.UTo(nil)
	S := svd.Values(nil)

	SMax, err := ArrayMax(S)
	if err != nil {
		panic(err)
	}

	singular_cutoff := 1e-15 * SMax

	for i := range S {
		if S[i] > singular_cutoff {
			S[i] = 1.0 / S[i]
		} else {
			S[i] = 0.0
		}
	}
	Ut := U.T()

	S_diag := mat.NewDiagDense(len(S), S)

	A := new(mat.Dense)
	A.Mul(S_diag, Ut)

	A.Mul(V, A)
	return A
}
