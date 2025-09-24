package variable

import "github.com/itsubaki/autograd/tensor"

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

	y := tensor.MatMul(x[0].Data, x[1].Data)
	return []*Variable{
		From(y),
	}
}

func (f *MatMulT) Backward(gy ...*Variable) []*Variable {
	gx := MatMul(gy[0], TransposeMatMul(f.w.NumDims())(f.w)) // gy * w.T
	gw := MatMul(TransposeMatMul(f.x.NumDims())(f.x), gy[0]) // x.T * gy

	if !tensor.ShapeEqual(gx.Shape(), f.x.Shape()) {
		gx = SumTo(f.x.Shape()...)(gx)
	}

	if !tensor.ShapeEqual(gw.Shape(), f.w.Shape()) {
		gw = SumTo(f.w.Shape()...)(gw)
	}

	return []*Variable{
		gx,
		gw,
	}
}
