package optimizer

import (
	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/model"
)

var (
	_ Model = (*model.MLP)(nil)
)

type Model interface {
	Params() []layer.Parameter
}
