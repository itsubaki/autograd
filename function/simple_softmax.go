package function

import "github.com/itsubaki/autograd/variable"

func SoftmaxSimple(x *variable.Variable) *variable.Variable {
	y := Exp(x)
	sumy := SumTo(len(y.Data), 1)(y)
	return Div(y, sumy)
}
