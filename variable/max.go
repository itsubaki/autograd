package variable

import (
	"math"

	"github.com/itsubaki/autograd/matrix"
)

func Max(x ...*Variable) *Variable {
	return (&Function{Forwarder: &MaxT{}}).ApplyAndFirst(x...)
}

type MaxT struct {
	x, y *Variable
}

func (f *MaxT) Forward(x ...*Variable) []*Variable {
	f.x = x[0]

	f.y = NewOf(matrix.Max(x[0].Data)...)
	return []*Variable{
		f.y,
	}
}

func (f *MaxT) Backward(gy ...*Variable) []*Variable {
	ybr := matrix.BroadcastTo(f.x.Shape(), f.y.Data)
	cond := NewOf(matrix.F2(f.x.Data, ybr, cond)...)
	gybr := BroadcastTo(cond.Shape()...)(gy[0])

	gx := Mul(gybr, cond) // argmax(gy)
	return []*Variable{
		gx,
	}
}

func cond(a, b float64) float64 {
	if math.Abs(a-b) < 1e-13 {
		return 1.0
	}

	return 0.0
}
