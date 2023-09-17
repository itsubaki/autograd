package numerical

import (
	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

type Func func(x ...*variable.Variable) []*variable.Variable

var (
	_ Func = F.Square
	_ Func = F.Exp
)

func Diff(f Func, x []*variable.Variable, h ...float64) []*variable.Variable {
	if len(h) == 0 {
		h = append(h, 1e-4)
	}

	x0 := make([]*variable.Variable, len(x))
	x1 := make([]*variable.Variable, len(x))
	for i := range x {
		x0[i] = variable.New(vector.AddC(x[i].Data, h[0])...)
		x1[i] = variable.New(vector.SubC(x[i].Data, h[0])...)
	}

	y0 := f(x0...)                                                     // f(x+h)
	y1 := f(x1...)                                                     // f(x-h)
	diff := func(a, b float64) float64 { return (a - b) / (2 * h[0]) } // (f(x+h) - f(x-h)) / (2*h)

	out := make([]*variable.Variable, len(y0))
	for i := range y0 {
		out[i] = &variable.Variable{Data: vector.F2(y0[i].Data, y1[i].Data, diff)}
	}

	return out
}
