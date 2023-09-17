package numerical

import (
	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

type Func func(x ...*variable.Variable) *variable.Variable

var (
	_ Func = F.Add
	_ Func = F.Exp
	_ Func = F.Square
)

func Diff(f Func, x []*variable.Variable, h ...float64) *variable.Variable {
	if len(h) == 0 {
		h = append(h, 1e-4)
	}

	y0 := f(add(x, h[0])...)                      // f(x+h)
	y1 := f(sub(x, h[0])...)                      // f(x-h)
	df := vector.F2(y0.Data, y1.Data, diff(h[0])) // (f(x+h) - f(x-h)) / 2h

	return &variable.Variable{Data: df}
}

func diff(h float64) func(a, b float64) float64 {
	return func(a, b float64) float64 { return (a - b) / (2 * h) }
}

func add(x []*variable.Variable, h float64) []*variable.Variable {
	x0 := make([]*variable.Variable, len(x))
	for i := range x {
		x0[i] = variable.New(vector.AddC(x[i].Data, h)...)
	}

	return x0
}

func sub(x []*variable.Variable, h float64) []*variable.Variable {
	x1 := make([]*variable.Variable, len(x))
	for i := range x {
		x1[i] = variable.New(vector.SubC(x[i].Data, h)...)
	}

	return x1
}
