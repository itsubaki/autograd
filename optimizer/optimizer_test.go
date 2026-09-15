package optimizer_test

import (
	"github.com/itsubaki/autograd/layer"
)

type TestModel struct {
	P layer.Parameter
}

func (m *TestModel) Params() layer.Parameters {
	return map[string]layer.Parameter{
		"p": m.P,
	}
}
