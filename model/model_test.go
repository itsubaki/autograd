package model_test

import (
	"testing"

	"github.com/itsubaki/autograd/model"
)

func TestModel_Add(t *testing.T) {
	m := &model.Model{}
	m.Add("linear", nil)

	defer func() {
		if r := recover(); r != nil {
			return
		}

		t.Fail() // unreachable
	}()

	m.Add("linear", nil)
	t.Fail() // unreachable
}
