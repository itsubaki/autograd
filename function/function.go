package function

import (
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

type Forwarder interface {
	Forward(x variable.Data) variable.Data
	Backward(gy variable.Data) variable.Data
	String() string
}

type Function struct {
	X, Y    variable.Data
	in, out *variable.Variable
	Forwarder
}

func (f *Function) Input() *variable.Variable {
	return f.in
}

func (f *Function) Output() *variable.Variable {
	return f.out
}

func (f *Function) Apply(x *variable.Variable) *variable.Variable {
	f.X, f.Y = x.Data, f.Forward(x.Data)
	f.in, f.out = x, &variable.Variable{Data: f.Y, Creator: f}
	return f.out
}

func NumericalDiff(f func(x *variable.Variable) *variable.Variable, x *variable.Variable, eps ...float64) *variable.Variable {
	if len(eps) == 0 {
		eps = append(eps, 1e-4)
	}

	y0 := f(variable.New(vector.AddC(x.Data, eps[0])...))
	y1 := f(variable.New(vector.SubC(x.Data, eps[0])...))
	diff := vector.F2(y0.Data, y1.Data, func(a, b float64) float64 { return (a - b) / (2 * eps[0]) })

	return &variable.Variable{Data: diff}
}
