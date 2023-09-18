package function

import (
	"github.com/itsubaki/autograd/variable"
)

func Square(x ...*variable.Variable) *variable.Variable {
	return (&Function{Forwarder: &PowT{C: 2.0}}).Apply(x...)[0]
}
