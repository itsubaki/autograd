package function

import (
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

func Sub(x ...*variable.Variable) *variable.Variable {
	return (&Function{Forwarder: &SubT{}}).Apply(x...)[0]
}

type SubT struct{}

func (f *SubT) Forward(x ...*variable.Variable) []*variable.Variable {
	y := vector.Sub(x[0].Data, x[1].Data)
	return []*variable.Variable{
		variable.New(y...),
	}
}

func (f *SubT) Backward(gy ...*variable.Variable) []*variable.Variable {
	c := variable.NewLikeWith(-1.0, gy[0])
	return []*variable.Variable{
		gy[0],
		Mul(c, gy[0]),
	}
}
