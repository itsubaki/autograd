package function

import "github.com/itsubaki/autograd/variable"

func SigmoidSimple(x *variable.Variable) *variable.Variable {
	return DivC(1.0, AddC(1.0, Exp(Neg(x)))) // y = 1 / (1 + exp(-x))
}
