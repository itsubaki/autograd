package function

import (
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

func Sub(x ...*variable.Variable) *variable.Variable {
	return (&Function{Forwarder: &SubT{}}).Apply(x...)[0]
}

type SubT struct{}

func (f *SubT) Forward(x ...variable.Data) []variable.Data {
	y := vector.Sub(x[0], x[1])
	return []variable.Data{y}
}

func (f *SubT) Backward(gy ...*variable.Variable) []*variable.Variable {
	return []*variable.Variable{
		gy[0],
		variable.New(vector.MulC(gy[0].Data, -1.0)...),
	}
}
