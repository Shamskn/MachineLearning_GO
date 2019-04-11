package regression

import (
	"MachineLearning_GO/utils"
	"gonum.org/v1/gonum/mat"
)

type RidgeRegression struct {
	LinearRegression
	alpha float64
}

func NewRidgeRegression(fitIntercept bool, alpha float64) *RidgeRegression {
	return &RidgeRegression{
		LinearRegression: LinearRegression{
			fitIntercept: fitIntercept,
		},
		alpha: alpha,
	}
}

func (rr *RidgeRegression) Fit(X *mat.Dense, t *mat.VecDense) {
	rows, cols := X.Dims()
	rr.features = cols
	if rr.fitIntercept {
		// Building a column vector
		ones := utils.Ones(rows)

		X_wBias := new(mat.Dense)
		X_wBias.Augment(ones, X)
		X = X_wBias
		cols++
	}

	ones := utils.Ones(cols)
	ones.ScaleVec(rr.alpha, ones)

	id_mtrx := mat.NewDiagDense(cols, ones.RawVector().Data)
	x_add := new(mat.Dense)
	x_add.Mul(X.T(), X)
	x_add.Add(id_mtrx, x_add)

	weight := new(mat.Dense)
	weight.Mul(X.T(), t)

	if err := weight.Solve(x_add, weight); err != nil {
		panic(err)
	}

	rr.weights = weight.RawMatrix().Data
	rr.isFitted = true
}
