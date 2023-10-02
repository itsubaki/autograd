package variable

import (
	"github.com/itsubaki/autograd/vector"
)

func Div(x ...*Variable) *Variable {
	return (&Function{Forwarder: &DivT{}}).ApplyAndFirst(x...)
}

type DivT struct {
	x0, x1           *Variable
	x0Shape, x1Shape []int
}

func (f *DivT) Forward(x ...*Variable) []*Variable {
	f.x0Shape, f.x1Shape = vector.Shape(x[0].Data), vector.Shape(x[1].Data)
	f.x0, f.x1 = x[0], x[1]

	x0, x1 := vector.Broadcast(x[0].Data, x[1].Data)
	y := vector.Div(x0, x1)
	return []*Variable{
		New(y...),
	}
}

func (f *DivT) Backward(gy ...*Variable) []*Variable {
	gx0 := Div(gy[0], f.x1)
	gx1 := Mul(gy[0], Div(Neg(f.x0), Mul(f.x1, f.x1))) // gy * (-x0 / x1^2)

	if vector.Equals(f.x0Shape, f.x1Shape) {
		return []*Variable{
			gx0,
			gx1,
		}
	}

	return []*Variable{
		SumTo(f.x0Shape...)(gx0),
		SumTo(f.x1Shape...)(gx1),
	}
}
