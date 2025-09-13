package variable

import (
	"github.com/itsubaki/autograd/matrix"
)

func MatMul(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &MatMulT{},
	}).First(x...)
}

type MatMulT struct {
	x, w *Variable
}

func (f *MatMulT) Forward(x ...*Variable) []*Variable {
	f.x, f.w = x[0], x[1]

	y := matrix.MatMul(x[0].Data, x[1].Data)
	return []*Variable{
		NewFrom(y),
	}
}

func (f *MatMulT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		MatMul(gy[0], Transpose(f.w)), // gy * w.T
		MatMul(Transpose(f.x), gy[0]), // x.T * gy
	}
}
