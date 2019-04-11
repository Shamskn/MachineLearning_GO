package regression

import (
	"MachineLearning_GO/utils"
	"gonum.org/v1/gonum/mat"
)

type LinearRegression struct {
	Regression
	weights      []float64
	fitIntercept bool
	features     int
	isFitted     bool
}

func NewLinearRegression(fitIntercept bool) *LinearRegression {
	return &LinearRegression{
		fitIntercept: fitIntercept,
	}
}

func (lr *LinearRegression) Fit(X *mat.Dense, t *mat.VecDense) {
	//w = np.inv(X.T @ X) @ X.T @ t
	rows, cols := X.Dims()
	lr.features = cols
	if lr.fitIntercept {
		// Building a column vector
		ones := utils.Ones(rows)

		X_wBias := new(mat.Dense)
		X_wBias.Augment(ones, X)
		X = X_wBias
	}
	x_inv := new(mat.Dense)
	x_inv.Mul(X.T(), X)

	//if err := x_inv.Inverse(x_inv); err != nil {
	//	print(err)
	//}
	// calculating the moore penrose pseudo inverse
	pinv := utils.Pinv(x_inv)

	weight := new(mat.Dense)
	weight.Product(pinv, X.T(), t)

	lr.weights = weight.RawMatrix().Data
	lr.isFitted = true
}

func (lr *LinearRegression) Predict(X *mat.Dense) *mat.VecDense {
	//y = self.w[0] + X @ self.w[1:] if self.fit_intercept else X @ self.w
	if !lr.isFitted {
		panic("Data not yet fitted")
	}
	rows, _ := X.Dims()
	pred := new(mat.VecDense)
	if lr.fitIntercept {
		bias := lr.weights[0]
		coef := mat.NewVecDense(lr.features, lr.weights[1:])

		pred.MulVec(X, coef)
		pred.AddScaledVec(pred, bias, utils.Ones(rows))

	} else {
		pred.MulVec(X, mat.NewVecDense(lr.features, lr.weights))
	}
	return pred
}
