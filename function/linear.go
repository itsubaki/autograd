package function

import (
	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"
)

func Linear(x ...*variable.Variable) *variable.Variable {
	return (&variable.Function{Forwarder: &LinearT{}}).ApplyAndFirst(x...)
}

type LinearT struct {
	x, w, b *variable.Variable
}

func (f *LinearT) Forward(x ...*variable.Variable) []*variable.Variable {
	f.x, f.w = x[0], x[1]

	y := matrix.Dot(x[0].Data, x[1].Data)
	if len(x) < 3 {
		// no bias
		return []*variable.Variable{
			variable.NewOf(y...),
		}
	}

	// add bias
	f.b, y = x[2], matrix.Add(y, x[2].Data)
	return []*variable.Variable{
		variable.NewOf(y...),
	}
}

func (f *LinearT) Backward(gy ...*variable.Variable) []*variable.Variable {
	gxs := []*variable.Variable{
		MatMul(gy[0], Transpose(f.w)), // gy * w.T
		MatMul(Transpose(f.x), gy[0]), // x.T * gy
	}
	if f.b == nil {
		// no bias
		return gxs
	}

	// add bias
	gb := SumTo(variable.Shape(f.b)...)(gy[0])
	return append(gxs, gb)
}
