package function

import (
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

func Cos(x ...*variable.Variable) *variable.Variable {
	return (&Function{Forwarder: &CosT{}}).Apply(x...)[0]
}

type CosT struct {
	x *variable.Variable
}

func (f *CosT) Forward(x ...*variable.Variable) []*variable.Variable {
	f.x = x[0]

	y := vector.Cos(f.x.Data)
	return []*variable.Variable{
		variable.New(y...),
	}
}

func (f *CosT) Backward(gy ...*variable.Variable) []*variable.Variable {
	return []*variable.Variable{
		Mul(Sin(f.x), gy[0]).MulC(-1),
	}
}
