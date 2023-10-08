package function

import "github.com/itsubaki/autograd/variable"

func MeanSquaredErrorSimple(x0, x1 *variable.Variable) *variable.Variable {
	diff := Sub(x0, x1)
	N := float64(len(diff.Data))
	return MulC(1.0/N, Sum(Mul(diff, diff)))
}
