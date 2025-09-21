package variable

import "github.com/itsubaki/autograd/tensor"

func MulC(c float64, x ...*Variable) *Variable {
	return (&Function{Forwarder: &MulT{}}).First(New(c), x[0])
}

func Mul(x ...*Variable) *Variable {
	return (&Function{Forwarder: &MulT{}}).First(x...)
}

type MulT struct {
	x0, x1           *Variable
	x0Shape, x1Shape []int
}

func (f *MulT) Forward(x ...*Variable) []*Variable {
	f.x0, f.x1 = x[0], x[1]
	f.x0Shape, f.x1Shape = x[0].Shape(), x[1].Shape()

	y := tensor.Mul(x[0].Data, x[1].Data)
	return []*Variable{
		From(y),
	}
}

func (f *MulT) Backward(gy ...*Variable) []*Variable {
	gx0 := Mul(gy[0], f.x1) // gy * x1
	gx1 := Mul(gy[0], f.x0) // gy * x0

	if equal(f.x0Shape, f.x1Shape) {
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
