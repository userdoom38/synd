"""
MIT License

Copyright (c) 2023 Wilhelm Ågren

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

File created: 2023-05-10
Last updated: 2023-05-10
"""

class Dataset(object):
    """ Base dataset class for single- or multi-table databases. """

    def fit(self, *args, **kwargs):
        raise NotImplementedError(
            f'No data fitting method is implemented for {self.__class__}. ',
            f'Please implement fitting a `DataTransformer` with data, transform ',
            f'the data, and create a `DataSampler` with the transformed data.',
        )

    def is_fitted(self, *args, **kwargs):
        raise NotImplementedError(
            f'No logic implemented to verify if `DataTransformer` has already been fitted. ',
            f'Please implement this, e.g., in a `getattr` was as for `SingleTable`.',
        )

