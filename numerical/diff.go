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

func Diff(f Func, x *variable.Variable, h ...float64) []*variable.Variable {
	if len(h) == 0 {
		h = append(h, 1e-4)
	}

	y0 := f(variable.New(vector.AddC(x.Data, h[0])...))                // f(x+h)
	y1 := f(variable.New(vector.SubC(x.Data, h[0])...))                // f(x-h)
	diff := func(a, b float64) float64 { return (a - b) / (2 * h[0]) } // (f(x+h) - f(x-h)) / (2*h)

	out := make([]*variable.Variable, len(y0))
	for i := range y0 {
		out[i] = &variable.Variable{Data: vector.F2(y0[i].Data, y1[i].Data, diff)}
	}

	return out
}
