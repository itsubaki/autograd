package function

import "github.com/itsubaki/autograd/variable"

func MeanSquaredErrorSimple(x0, x1 *variable.Variable) *variable.Variable {
	diff := Sub(x0, x1)
	return MulC(1.0/float64(diff.N()), Sum(Mul(diff, diff))) // (1/N) * sum((x0 - x1)^2)
}
