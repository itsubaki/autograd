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
	if len(x) > 2 {
		f.b = x[2] // bias
		y = matrix.Add(y, matrix.BroadcastTo(matrix.Shape(y), f.b.Data))
	}

	return []*variable.Variable{
		variable.NewOf(y...),
	}
}

func (f *LinearT) Backward(gy ...*variable.Variable) []*variable.Variable {
	gxs := []*variable.Variable{
		MatMul(gy[0], Transpose(f.w)), // gy * w.T
		MatMul(Transpose(f.x), gy[0]), // x.T * gy
	}

	if f.b != nil {
		gxs = append(gxs, SumTo(f.b.Shape()...)(gy[0]))
	}

	return gxs
}
