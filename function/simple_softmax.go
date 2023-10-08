package function

import (
	"github.com/itsubaki/autograd/variable"
)

func SoftmaxSimple(x *variable.Variable) *variable.Variable {
	y := Exp(x)                       // (N, M)
	sumy := SumTo(y.Shape()[0], 1)(y) // (N, 1)
	return Div(y, sumy)
}
