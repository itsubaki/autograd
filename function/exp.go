package function

import (
	"math"

	"github.com/itsubaki/autograd/variable"
)

func Exp(x *variable.Variable) *variable.Variable {
	return (&ExpT{}).Apply(x)
}

type ExpT struct {
	X, Y    variable.Data
	in, out *variable.Variable
}

func (f *ExpT) Input() *variable.Variable {
	return f.in
}

func (f *ExpT) Output() *variable.Variable {
	return f.out
}

func (f *ExpT) Apply(x *variable.Variable) *variable.Variable {
	f.X, f.Y = x.Data, f.Forward(x.Data)
	f.in, f.out = x, &variable.Variable{Data: f.Y, Creator: f}
	return f.out
}

func (f *ExpT) Forward(x variable.Data) variable.Data {
	y := variable.NewData(len(x))
	for i := 0; i < len(x); i++ {
		y[i] = math.Exp(x[i])
	}

	return y
}

func (f *ExpT) Backward(gy variable.Data) variable.Data {
	x := f.in.Data

	grad := variable.NewData(len(x))
	for i := 0; i < len(x); i++ {
		grad[i] = math.Exp(x[i]) * gy[i]
	}

	return grad
}
