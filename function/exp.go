package function

import (
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

func Exp(x ...*variable.Variable) *variable.Variable {
	return (&Function{Forwarder: &ExpT{}}).Apply(x...)[0]
}

type ExpT struct {
	x *variable.Variable
}

func (f *ExpT) Forward(x ...*variable.Variable) []*variable.Variable {
	f.x = x[0]

	y := vector.Exp(f.x.Data)
	return []*variable.Variable{
		variable.New(y...),
	}
}

func (f *ExpT) Backward(gy ...*variable.Variable) []*variable.Variable {
	return []*variable.Variable{
		Mul(Exp(f.x), gy[0]), // exp(x) * gy
	}
}
