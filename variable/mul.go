package variable

import (
	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/vector"
)

func MulC(c float64, x ...*Variable) *Variable {
	return (&Function{Forwarder: &MulT{}}).ApplyAndFirst(Const(c), x[0])
}

func Mul(x ...*Variable) *Variable {
	return (&Function{Forwarder: &MulT{}}).ApplyAndFirst(x...)
}

type MulT struct {
	x0, x1           *Variable
	x0Shape, x1Shape []int
}

func (f *MulT) Forward(x ...*Variable) []*Variable {
	f.x0Shape, f.x1Shape = x[0].Shape(), x[1].Shape()
	f.x0, f.x1 = x[0], x[1]

	x0, x1 := matrix.Broadcast(x[0].Data, x[1].Data)
	y := matrix.Mul(x0, x1)
	return []*Variable{
		NewOf(y...),
	}
}

func (f *MulT) Backward(gy ...*Variable) []*Variable {
	gx0 := Mul(gy[0], f.x1) // gy * x1
	gx1 := Mul(gy[0], f.x0) // gy * x0

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
