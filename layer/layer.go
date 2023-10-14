package layer

import "github.com/itsubaki/autograd/variable"

type Forwarder interface {
	Forward(x ...*variable.Variable) []*variable.Variable
	Params() []Parameter
}

type Layer struct {
	Input, Output []*variable.Variable
	Layers        []*Layer
	Forwarder
}

func (l *Layer) First(x ...*variable.Variable) *variable.Variable {
	return l.Forward(x...)[0]
}

func (l *Layer) Forward(x ...*variable.Variable) []*variable.Variable {
	y := l.Forwarder.Forward(x...)
	l.Input, l.Output = x, y
	return y
}

func (l *Layer) Add(layer *Layer) {
	l.Layers = append(l.Layers, layer)
}

func (l *Layer) Params() []Parameter {
	params := make([]Parameter, 0)
	if l.Forwarder != nil {
		params = l.Forwarder.Params()
	}

	for _, ll := range l.Layers {
		params = append(params, ll.Params()...)
	}

	return params
}

func (l *Layer) Cleargrads() {
	for _, p := range l.Params() {
		p.Cleargrad()
	}
}
