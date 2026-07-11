package tensor_test

import (
	"fmt"
	"testing"

	"github.com/itsubaki/autograd/tensor"
)

func ExampleIterator() {
	a := tensor.New([]int{2, 3}, []int{
		1, 2, 3,
		4, 5, 6,
	})

	b := tensor.New([]int{2, 3}, []int{
		10, 20, 30,
		40, 50, 60,
	})

	a = tensor.Transpose(a)
	b = tensor.Transpose(b)

	it := tensor.NewIterator(a, b)
	for it.Next() {
		ia, ib := it.Offset(0), it.Offset(1)
		fmt.Println(ia, ib, ":", a.Data[ia], b.Data[ib])
	}

	// Output:
	// 0 0 : 1 10
	// 3 3 : 4 40
	// 1 1 : 2 20
	// 4 4 : 5 50
	// 2 2 : 3 30
	// 5 5 : 6 60
}

func ExampleIterator_scalar() {
	a := tensor.Scalar(1.0)
	it := tensor.NewIterator(a)
	for it.Next() {
		ia := it.Offset(0)
		fmt.Println(ia, ":", a.Data[ia])
	}

	// Output:
	// 0 : 1
}

func ExampleIterator_broadcast() {
	a := tensor.New([]int{1, 3}, []int{
		1, 2, 3,
	})
	a = tensor.BroadcastTo(a, 2, 3)

	it := tensor.NewIterator(a)
	for it.Next() {
		i := it.Offset(0)
		fmt.Println(i, ":", a.Data[i])
	}

	// Output:
	// 0 : 1
	// 1 : 2
	// 2 : 3
	// 0 : 1
	// 1 : 2
	// 2 : 3
}

func ExampleIterator_done() {
	a := tensor.New([]int{1, 3}, []int{
		1, 2, 3,
	})

	it := tensor.NewIterator(a)
	fmt.Println(it.Next())
	fmt.Println(it.Next())
	fmt.Println(it.Next())
	fmt.Println(it.Next())
	fmt.Println(it.Next())

	// Output:
	// true
	// true
	// true
	// false
	// false
}

func TestIterator_shape(t *testing.T) {
	a := tensor.New([]int{2, 3}, []int{
		1, 2, 3,
		4, 5, 6,
	})

	b := tensor.New([]int{2, 3}, []int{
		10, 20, 30,
		40, 50, 60,
	})

	func() {
		defer func() {
			if r := recover(); r != nil {
				return
			}

			t.Errorf("unexpected panic for shape %v and %v", a.Shape(), b.Shape())
		}()

		a = tensor.Transpose(a)
		_ = tensor.NewIterator(a, b)
		t.Fail()
	}()
}
