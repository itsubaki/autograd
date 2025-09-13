package function

import "github.com/itsubaki/autograd/variable"

func SoftmaxSimple(x *variable.Variable) *variable.Variable {
	y := Exp(x)
	N := x.Shape()[0]
	sumy := SumTo(N, 1)(y)
	return Div(y, sumy)
}
