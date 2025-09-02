import { increment, decrement } from "./redux/slices/counterSlice";
import { useAppDispatch, useAppSelector } from "./redux/hooks";
import { useGetUsersQuery } from "./redux/services/api";

function App() {
    const count = useAppSelector((state) => state.counter.value);
    const dispatch = useAppDispatch();
    const { data: users, isLoading } = useGetUsersQuery();

    return (
        <div className="h-screen flex flex-col items-center justify-center bg-gray-100">
            <h1 className="text-3xl font-bold mb-4">MERN + TS + Tailwind + Redux</h1>

            <p className="text-xl mb-4">Count: {count}</p>
            <div className="space-x-4">
                <button
                    onClick={() => dispatch(increment())}
                    className="px-4 py-2 bg-blue-500 text-white rounded-lg shadow"
                >
                    Increment
                </button>
                <button
                    onClick={() => dispatch(decrement())}
                    className="px-4 py-2 bg-red-500 text-white rounded-lg shadow"
                >
                    Decrement
                </button>
            </div>

            <div className="mt-6">
                {isLoading ? (
                    <p>Loading users...</p>
                ) : (
                    <ul className="list-disc">
                        {users?.map((u, i) => (
                            <li key={i}>{u.name}</li>
                        ))}
                    </ul>
                )}
            </div>
        </div>
    );
}

export default App;
