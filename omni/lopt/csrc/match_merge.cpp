// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
using std::vector;
namespace py = pybind11;

template <typename T> vector<T> ndarray_to_vector(py::array_t<T> input_array) {
    py::buffer_info buf = input_array.request();
    T *ptr = static_cast<T *>(buf.ptr);
    return vector<T>(ptr, ptr + buf.size); 
}
template <typename T> py::array_t<T> vector_to_ndarray(vector<T> input_array) {
    return py::array_t<T>(input_array.size(), input_array.data());
}
template <typename T> py::tuple vector_to_PyTuple(vector<T> input_vector) {
    py::tuple res(input_vector.size());
    size_t len = input_vector.size();
    for (size_t i = 0; i < len; i++)
        res[i] = input_vector[i];
    return res;
}

#include <iostream>
using std::cout;

template <typename T>
py::tuple match(py::array_t<T> _chunks0, py::array_t<T> _chunks1, int chunk_size, int Mismatch_thres) {
    auto chunks0 = ndarray_to_vector(_chunks0);
    auto chunks1 = ndarray_to_vector(_chunks1);

    int match_a = 0, match_b = 0, match_size = 0;
    int len0 = chunks0.size(), len1 = chunks1.size();
    int i0 = len0 - 1, current_match_len = 0;
    for (int i1 = len1 - 1; i1 >=0; i1--) {
        while ((i0 >= 0) && chunks0[i0] > chunks1[i1] + chunk_size) {
            i0--;
            current_match_len = 0;
        }
        if ((i0 >= 0) && chunks0[i0] == chunks1[i1] + chunk_size) {
            if (current_match_len > match_size) {
                match_a = i0;
                match_b = i1;
                match_size = current_match_len;
            }
            current_match_len++;
            i0--;
        }
        else
            current_match_len = 0;
    }
    if (match_size <= Mismatch_thres)
        return vector_to_PyTuple<int>({-1, -1});
    match_a = len0 - (match_a + match_size - 1);
    match_b = len1 - (match_b + match_size - 1);

    return vector_to_PyTuple<int>({match_a, match_b});
}


template <typename T>
py::array_t<T> merge(py::list _chunks, py::list _matches) {
    vector<vector<T> > chunks;
    vector<T> result;
    for (auto item : _chunks)
        chunks.push_back(ndarray_to_vector(item.cast<py::array_t<T> >()));

    if (_matches.size() != chunks.size() * 2){
        cout << "_matches.size() != chunks.size() * 2\n";
        return vector_to_ndarray<int64_t>({-1, -1});
    }
    for (int i = 0, n = chunks.size(); i < n; i++)
        result.insert(result.end(),
        chunks[i].end() - int(_matches[i<<1].cast<int>()),
        chunks[i].end() - int(_matches[i<<1|1].cast<int>()));
    return vector_to_ndarray(result);
}

PYBIND11_MODULE(Cpp_match_merge, m) {
    m.doc()= "cpp - token match and merge";
    m.def("match", [](py::array_t<int64_t> _chunks0, py::array_t<int64_t> _chunks1, int chunk_size, int Mismatch_thres) { return match<int64_t>(_chunks0, _chunks1, chunk_size, Mismatch_thres); },
        "int64: match Longest Common Substring between _chunks0 and _chunks1, requires them to be non-decreasing",
        py::arg("_chunks0"), py::arg("_chunks1"), py::arg("chunk_size"), py::arg("Mismatch_thres"));
    m.def("merge", [](py::list _chunks, py::list _matches) { return merge<int64_t>(_chunks, _matches); },
        "int64: merge the chunks using matches",
        py::arg("_chunks"), py::arg("_matches"));
}