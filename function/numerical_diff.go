package function

import (
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

func NumericalDiff(f func(x *variable.Variable) *variable.Variable, x *variable.Variable, h ...float64) *variable.Variable {
	if len(h) == 0 {
		h = append(h, 1e-4)
	}

	y0 := f(variable.New(vector.AddC(x.Data, h[0])...))                // f(x+h)
	y1 := f(variable.New(vector.SubC(x.Data, h[0])...))                // f(x-h)
	diff := func(a, b float64) float64 { return (a - b) / (2 * h[0]) } // (f(x+h) - f(x-h)) / (2*h)

	return &variable.Variable{Data: vector.F2(y0.Data, y1.Data, diff)}
}
