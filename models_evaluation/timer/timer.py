"""
This module aims to provide tools to easily time snippets of codes with color coding.
"""

import math
from datetime import datetime as dt
from time import time
from typing import Any

try:
    from colorama import Fore, Style, init

    init()
except ModuleNotFoundError:
    # Emulate the Fore and Style class of colorama with a class that as an empty string for every attributes.
    class EmptyStringAttrClass:

        def __getattr__(self, attr):
            return ''

    Fore = EmptyStringAttrClass()
    Style = EmptyStringAttrClass()

__author__ = 'Jean-Samuel Leboeuf, Frédérik Paradis'
__date__ = 'May 28th, 2019'


class Timer:
    """
    This class can be used to time snippets of code inside your code. It prints colored information in the terminal.

    The class can be used as a context manager to time the code inside the 'with' statement, as a decorator of a
    function or a method to time it at each call, or as an iterator to have the total running time of a
    for loop as well as the mean time taken per iteration. See the doc of the init method for usage examples.
    """

    def __init__(self,
                 func_or_name=None,
                 *,
                 display_name='',
                 datetime_format='%Y-%m-%d %Hh%Mm%Ss',
                 elapsed_time_format='short',
                 main_color='LIGHTYELLOW_EX',
                 exception_exit_color='LIGHTRED_EX',
                 name_color='LIGHTBLUE_EX',
                 time_color='LIGHTCYAN_EX',
                 datetime_color='LIGHTMAGENTA_EX',
                 yield_timer=False):
        """
        Args:
            func_or_name (Union[Callable, str, None]):
                If Timer is used as a decorator: If a callable, the callable will be wrapped and timed every time it is
                called. If None, the callable will be set on the next call to Timer.
                If Timer is used as a context manager: If a string, the string will be used as display name. If None,
                no name will be displayed.
            display_name (Union[str, None]):
                String to be displayed to identify the timed snippet of code. Default (an empty string) will display
                the name of the function. If set to None, will display nothing instead. Only useful if Timer
                is used as a decorator, since the first arguments is used in the case of a context manager.
            datetime_format (str or None, optional):
                Datetime format used to display the date and time. The format follows the template of the 'datetime'
                package. If None, no date or time will be displayed.
            elapsed_time_format (either 'short' or 'long', optional):
                Format used to display the elapsed time. If 'long', whole words will be used. If 'short', only the
                first letters will be displayed.
            main_color (str):
                Color in which the main text will be displayed. Choices are those from the package colorama.
            exception_exit_color (str):
                Color in which the exception text will be displayed. Choices are those from the package colorama.
            name_color (str):
                Color in which the function name will be displayed. Choices are those from the package colorama.
            time_color (str):
                Color in which the time taken by the function will be displayed. Choices are those from the package
                colorama.
            datetime_color (str):
                Color in which the date and time of day will be displayed. Choices are those from the package colorama.
            yield_timer (bool):
                Whether or not to yield the Timer object in addition to the output when Timer is used as an iterator.

        Supported colors:
            BLACK, WHITE, RED, BLUE, GREEN, CYAN, MAGENTA, YELLOW, LIGHTRED_EX, BLIGHTLUE_EX, GRLIGHTEEN_EX,
            CLIGHTYAN_EX, MAGELIGHTNTA_EX, YELLIGHTLOW_EX

        The class can be used as a context manager, a decorator or as an iterator.

        Examples as a context manager:
            Example 1:
                >>> from graal_utils import Timer
                >>> with Timer():
                ...     print('graal')
                ...
            Execution started on 2019-05-09 13h48m23s.

            graal

            Execution completed in 0.00 seconds on 2019-05-09 13h48m23s.

            Example 2:
                >>> from graal_utils import Timer
                >>> with Timer('python', time_color='MAGENTA'):
                ...     print('Python')
                ...
            Execution of 'python' started on 2019-05-09 13h48m23s.

            Python

            Execution of 'python' completed in 0.00 seconds on 2019-05-09 13h48m23s.

        Examples as a decorator:
            Example 1:
                >>> from graal_utils import Timer
                >>> @Timer
                ... def foo():
                ...     print('foo!')
                ...
                >>> foo()
                Execution of 'foo' started on 2018-09-10 20h25m06s.

                foo!

                Execution of 'foo' completed in 0.00 seconds on 2018-09-10 20h25m06s.

            Example 2:
                >>> @Timer(datetime_format='%Hh%Mm%Ss', display_func_name=False, main_color='WHITE')
                ... def bar():
                ...     print('bar!')
                ...     raise RuntimeError
                ...
                >>> try:
                ...     bar()
                ... except RuntimeError: pass
                Execution started on 20h25m06s.

                bar!

                Execution terminated after 0.00 seconds on 20h25m06s.

                >>> bar.elapsed_time
                0.5172324180603027

            Example 3:
                >>> class Spam:
                ...     @Timer
                ...     def spam(self):
                ...         print('egg!')

                >>> Spam().spam()
                Execution of 'spam' started on 2018-10-02 18h33m14s.

                egg!

                Execution of 'spam' completed in 0.00 seconds on 2018-10-02 18h33m14s.

        Examples as an iterator:
            Example 1: Simple case.
                >>> for i in Timer(range(3)):
                ...     time.sleep(.1)
                ...     print(i)
                Execution of 'range' started on 2021-04-23 15h09m30s.

                0
                1
                2

                Execution of 'range' completed in 0.33s on 2021-04-23 15h09m31s.
                Mean time per iteration: 0.11s ± 0.00s over 3 iterations.
                Iteration 0 was the shortest with 0.10s.
                Iteration 1 was the longest with 0.11s.

            Example 2: Case with the Timer objected yielded.
                >>> for i, t in Timer(range(3), yield_timer=True):
                ...     sleep(.1)
                ... print(t.laps)
                Execution of 'range' started on 2021-04-23 15h16m29s.


                Execution of 'range' completed in 0.34s on 2021-04-23 15h16m29s.
                Mean time per iteration: 0.11s ± 0.00s over 3 iterations.
                Iteration 1 was the shortest with 0.11s.
                Iteration 0 was the longest with 0.11s.

                [0.11200141906738281, 0.11099791526794434, 0.11100339889526367]
        """
        if isinstance(func_or_name, str):
            func, display_name = None, func_or_name
        else:
            func = func_or_name

        self._wrapped_func = self.wrap_function(func)
        self.display_name = display_name
        self.start_time = None
        self.elapsed_time = None
        self.datetime_format = datetime_format
        self.elapsed_time_format = elapsed_time_format

        self.main_color = getattr(Fore, main_color)
        self.exception_exit_color = getattr(Fore, exception_exit_color)
        self.name_color = getattr(Fore, name_color)
        self.time_color = getattr(Fore, time_color)
        self.datetime_color = getattr(Fore, datetime_color)

        self.yield_timer = yield_timer
        self.iter_stats = ''

    def __enter__(self):
        self._start_timer()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.elapsed_time = time() - self.start_time
        if exc_type:
            self._exception_exit_end_timer()
        else:
            self._normal_exit_end_timer()

    @property
    def func_name(self):
        # pylint: disable=no-else-return
        if self.display_name:
            return f"of '{self.name_color}{self.display_name}{self.main_color}' "
        elif self.display_name == '' and self._wrapped_func is not None:
            return f"of '{self.name_color}{self.__name__}{self.main_color}' "
        else:
            return ''

    @property
    def datetime(self):
        # pylint: disable=no-else-return
        if self.datetime_format is None:
            return ''
        else:
            return 'on ' + self.datetime_color + dt.now().strftime(self.datetime_format) + self.main_color

    @staticmethod
    def format_long_time(seconds, period):
        # pylint: disable=no-else-return
        periods = {'d': 'day', 'h': 'hour', 'm': 'minute', 's': 'second'}

        pluralize = lambda period_value: 's' if period_value > 1 else ''
        format_period_string = periods[period] + pluralize(seconds)
        if period != 's':
            return f"{int(seconds)} {format_period_string}"
        else:
            return f"{seconds:.2f} {format_period_string}"

    @staticmethod
    def format_short_time(seconds, period):
        # pylint: disable=no-else-return
        if period != 's':
            return f"{int(seconds)}{period}"
        else:
            return f"{seconds:.2f}{period}"

    def format_elapsed_time(self, seconds):
        is_long = self.elapsed_time_format == 'long'
        format_time = self.format_long_time if is_long else self.format_short_time
        periods = {
            'd': 60 * 60 * 24,
            'h': 60 * 60,
            'm': 60,
        }

        time_strings = []
        for period_name, period_seconds in periods.items():
            if seconds >= period_seconds:
                period_value, seconds = divmod(seconds, period_seconds)
                time_strings.append(format_time(period_value, period_name))

        time_strings.append(format_time(seconds, 's'))

        return self.time_color + " ".join(time_strings) + self.main_color

    def _start_timer(self):
        self.start_time = time()
        print(self.main_color + f'Execution {self.func_name}started {self.datetime}.\n' + Style.RESET_ALL)

    def _exception_exit_end_timer(self):
        print(self.exception_exit_color + '\nExecution terminated after ' +
              self.format_elapsed_time(self.elapsed_time) +
              f'{self.exception_exit_color} {self.datetime}{self.exception_exit_color}.\n{self.iter_stats}' +
              Style.RESET_ALL)

    def _normal_exit_end_timer(self):
        print(self.main_color + f'\nExecution {self.func_name}completed in ' +
              self.format_elapsed_time(self.elapsed_time) + f' {self.datetime}.\n{self.iter_stats}' + Style.RESET_ALL)

    def wrap_function(self, func):
        if func is not None:
            self.__doc__ = func.__doc__
            if hasattr(func, '__name__'):
                self.__name__ = func.__name__
            else:
                self.__name__ = type(func).__name__  # For the case when Timer is used as an iterator.

        return func

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # pylint: disable=no-else-return
        if self._wrapped_func is None:
            self._wrapped_func = self.wrap_function(args[0])
            return self
        else:
            with self:
                return self._wrapped_func(*args, **kwargs)

    def __get__(self, parent_of_wrapped_method, type_of_parent_of_wrapped_method=None):
        # Gets called only for wrapped methods. Sets the first argument of the function as the correct
        # instance of 'self'.
        self.__wrapped_method = self._wrapped_func
        self._wrapped_func = lambda *args, **kwargs: self.__wrapped_method(parent_of_wrapped_method, *args, **kwargs)
        return self

    def __iter__(self, *args, **kwargs):
        with self:
            self.laps = []
            try:
                for output in self._wrapped_func:
                    start_time = time()
                    yield (output, self) if self.yield_timer else output
                    self.laps.append(time() - start_time)
            finally:
                self._update_iter_stats()

    def _update_iter_stats(self):
        mean_time = sum(self.laps) / len(self.laps)
        std = math.sqrt(sum(t**2 for t in self.laps) / len(self.laps) - mean_time**2)
        shortest_time = min((t, i) for i, t in enumerate(self.laps))
        longest_time = max((t, i) for i, t in enumerate(self.laps))
        self.iter_stats = (
            self.main_color +
            f'Mean time per iteration: {self.format_elapsed_time(mean_time)} ± {self.format_elapsed_time(std)} over'
            f' {len(self.laps)} iterations.\n' +
            f'Iteration {shortest_time[1]} was the shortest with {self.format_elapsed_time(shortest_time[0])}.\n' +
            f'Iteration {longest_time[1]} was the longest with {self.format_elapsed_time(longest_time[0])}.\n')
