package function

import (
	"math"

	"github.com/itsubaki/autograd/variable"
)

func Square(x *variable.Variable) *variable.Variable {
	return (&SquareT{}).Apply(x)
}

type SquareT struct {
	X, Y    variable.Data
	in, out *variable.Variable
}

func (f *SquareT) Input() *variable.Variable {
	return f.in
}

func (f *SquareT) Output() *variable.Variable {
	return f.out
}

func (f *SquareT) Apply(x *variable.Variable) *variable.Variable {
	f.X, f.Y = x.Data, f.Forward(x.Data)
	f.in, f.out = x, &variable.Variable{Data: f.Y, Creator: f}
	return f.out
}

func (f *SquareT) Forward(x variable.Data) variable.Data {
	y := variable.NewData(len(x))
	for i := 0; i < len(x); i++ {
		y[i] = math.Pow(x[i], 2)
	}

	return y
}

func (f *SquareT) Backward(gy variable.Data) variable.Data {
	x := f.in.Data

	grad := variable.NewData(len(x))
	for i := 0; i < len(x); i++ {
		grad[i] = 2 * x[i] * gy[i] // 2x * gy
	}

	return grad
}
