package optimizer

import (
	"github.com/itsubaki/autograd/hook"
	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/model"
)

var (
	_ Model = (*model.MLP)(nil)
	_ Model = (*model.LSTM)(nil)
)

var (
	_ Hook = hook.WeightDecay(0.0)
)

type Model interface {
	Params() []layer.Parameter
}

type Hook func(params []layer.Parameter)
