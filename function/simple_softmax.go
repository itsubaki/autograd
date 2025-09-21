package function

import "github.com/itsubaki/autograd/variable"

func SoftmaxSimple(x *variable.Variable) *variable.Variable {
	return Div(Exp(x), SumTo(x.Shape()[0], 1)(Exp(x)))
}
