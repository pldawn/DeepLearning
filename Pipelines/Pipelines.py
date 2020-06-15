from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pprint import pprint
import itertools


class Pipelines:
    def __init__(self):
        self.pipelines = [{}]
        self.pipelines_iterator = None
        self.is_build = False
        self.allowed_process = []
        self.required_process = []
        self.process_method_nums = {}

    def add_process_method_num(self, process, num):
        self.process_method_nums[process] = num

    def add_process(self, process, method_indices):
        self.pipelines[0].update({process: method_indices})

    def add_sub_pipelines(self, sub_pipelines):
        self.pipelines.extend(sub_pipelines.pipelines)
        self.pipelines = [p for p in self.pipelines if p]

        self.update_process_method_nums(sub_pipelines.process_method_nums)

    def update_process_method_nums(self, new_process_method_num):
        for k, v in new_process_method_num.items():
            if k in self.process_method_nums:
                if v > self.process_method_nums[k]:
                    self.process_method_nums[k] = v
            else:
                self.process_method_nums.update({k: v})

    def check(self, sub_pipelines):
        if not sub_pipelines:
            return False

        for process in self.required_process:
            if process not in sub_pipelines:
                return False

        for process in sub_pipelines:
            if process not in self.allowed_process:
                return False

        return True

    def build(self):
        self.is_build = True

        # decode sub_pipelines
        invalid_sub_pipelines = []
        sub_pipelines_iterators = []
        for sub_pipelines in self.pipelines:
            if self.check(sub_pipelines):
                one_iterator = list(self.decode_sub_pipelines(sub_pipelines))
                sub_pipelines_iterators += one_iterator
            else:
                invalid_sub_pipelines.append(sub_pipelines)

        self.pipelines_iterator = iter(sub_pipelines_iterators)

        if len(invalid_sub_pipelines) > 0:
            pprint('Invalid pipelines: %d' % len(invalid_sub_pipelines))
            pprint(invalid_sub_pipelines)

    def decode_sub_pipelines(self, sub_pipelines):
        sub_pipelines_decode = {}

        for process, method_indices in sub_pipelines.items():
            if method_indices == 'all' or method_indices == 'ALL':
                sub_pipelines_decode[process] = [(process, value) for value in
                                                 list(range(self.process_method_nums[process]))]
            if type(method_indices) == list:
                sub_pipelines_decode[process] = [(process, value) for value in method_indices]
            if type(method_indices) == int:
                sub_pipelines_decode[process] = [(process, method_indices)]
            if type(method_indices) == tuple and len(method_indices) == 2:
                sub_pipelines_decode[process] = [(process, value) for value in
                                                 list(range(method_indices[0], method_indices[1] + 1))]

        one_iterator = itertools.product(*list(sub_pipelines_decode.values()))

        return one_iterator
