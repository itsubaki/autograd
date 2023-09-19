package function

import (
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

func Sin(x ...*variable.Variable) *variable.Variable {
	return (&Function{Forwarder: &SinT{}}).ApplyS(x...)
}

type SinT struct {
	x *variable.Variable
}

func (f *SinT) Forward(x ...*variable.Variable) []*variable.Variable {
	f.x = x[0]

	y := vector.Sin(f.x.Data)
	return []*variable.Variable{
		variable.New(y...),
	}
}

func (f *SinT) Backward(gy ...*variable.Variable) []*variable.Variable {
	return []*variable.Variable{
		Mul(Cos(f.x), gy[0]), // cos(x) * gy
	}
}
