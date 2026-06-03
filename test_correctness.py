from problem import ProblemInstance
from greedy import greedy_solve
from genetic import genetic_solve


def test_objective_repeats():
    p = ProblemInstance(
        n=1, k=2, alpha=0.5,
        a=[10], t=[1], o=[0, 0], c=[999, 999],
        T=[10, 10], s=[0, 0], m=[1, 1],
        d=[[0, 1], [1, 0]]
    )
    assert p.objective([[1], [1]]) == 15.0, p.objective([[1], [1]])


def test_time_budget_with_start_offset():
    p = ProblemInstance(
        n=1, k=1, alpha=0.5,
        a=[10], t=[1], o=[0, 0], c=[999, 999],
        T=[6], s=[9], m=[1],
        d=[[0, 1], [1, 0]]
    )
    assert p.route_feasible([1], 0)


def test_no_duplicates_inside_route():
    p = ProblemInstance(
        n=1, k=1, alpha=0.5,
        a=[10], t=[1], o=[0, 0], c=[999, 999],
        T=[10], s=[0], m=[2],
        d=[[0, 1], [1, 0]]
    )
    assert not p.route_feasible([1, 1], 0)


def test_example_5_greedy():
    p = ProblemInstance(
        n=5, k=2, alpha=0.7,
        a=[8, 6, 7, 9, 5],
        t=[30, 20, 25, 35, 15],
        o=[0, 0, 60, 0, 120, 0],
        c=[999, 480, 540, 420, 540, 660],
        T=[120, 120], s=[0, 0], m=[3, 3],
        d=[
            [0,15,20,10,25,12],
            [15,0,10,12,15,8],
            [20,10,0,14,10,9],
            [10,12,14,0,16,7],
            [25,15,10,16,0,11],
            [12,8,9,7,11,0],
        ]
    )
    res = greedy_solve(p)
    print('greedy routes', res.routes, res.objective)
    assert res.routes == [[3,5,1], [3,5,2]], res.routes
    assert abs(res.objective - 34.4) < 1e-6, res.objective


if __name__ == '__main__':
    test_objective_repeats()
    test_time_budget_with_start_offset()
    test_no_duplicates_inside_route()
    test_example_5_greedy()
    print('all tests passed')
