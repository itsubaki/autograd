package layer

import "github.com/itsubaki/autograd/variable"

type Forwarder interface {
	Forward(x ...*variable.Variable) []*variable.Variable
	Params() map[string]Parameter
}

type Layer struct {
	Input, Output []*variable.Variable
	Forwarder
}

func (l *Layer) ApplyAndFirst(x ...*variable.Variable) *variable.Variable {
	return l.Apply(x...)[0]
}

func (l *Layer) Apply(x ...*variable.Variable) []*variable.Variable {
	y := l.Forward(x...)
	l.Input, l.Output = x, y
	return y
}

func (l *Layer) Cleargrads() {
	for _, p := range l.Params() {
		p.Cleargrad()
	}
}
