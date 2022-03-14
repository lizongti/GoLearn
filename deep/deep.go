package deep

import "gorgonia.org/tensor"

func Run() {

}

func coordinates(dt tensor.Dtype) (x tensor.Tensor, y tensor.Tensor) {
	var xBack, yBack interface{}
	switch dt {
	case tensor.Float32:
	case tensor.Float64:
	default:
		panic("unsupport type")
	}
}
