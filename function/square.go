package function

import (
	"github.com/itsubaki/autograd/variable"
)

func Square(x ...*variable.Variable) *variable.Variable {
	return (&Function{Forwarder: &SquareT{PowT{C: 2.0}}}).ApplyS(x...)
}

type SquareT struct {
	PowT
}
