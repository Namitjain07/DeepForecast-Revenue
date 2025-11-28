import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import AdminNavbar from '../components/dashboard/AdminNavbar';
import HotelCard from '../components/dashboard/HotelCard';
import { useAppDispatch, useAppSelector } from '../redux/hooks';
import { fetchAllHotels, searchHotels } from '../redux/services/api';
import { clearSearch } from '../redux/slices/hotelSlice';

const AllHotels: React.FC = () => {
    const role = localStorage.getItem('userRole') as 'admin' | 'owner' | 'manager' || 'admin';
    const navigate = useNavigate();
    const dispatch = useAppDispatch();

    const { hotels, pagination, loading, searchResults, searchPagination } = useAppSelector((state) => state.hotels);

    const [searchTerm, setSearchTerm] = useState('');
    const [currentPage, setCurrentPage] = useState(1);
    const [hasMore, setHasMore] = useState(true);
    const scrollContainerRef = useRef<HTMLDivElement>(null);
    const observerRef = useRef<IntersectionObserver | null>(null);

    // Fetch all hotels on mount
    useEffect(() => {
        dispatch(fetchAllHotels(1, 10) as any);
    }, [dispatch]);

    // Handle infinite scroll
    useEffect(() => {
        const observerElement = scrollContainerRef.current;
        if (!observerElement) return;

        observerRef.current = new IntersectionObserver(
            (entries) => {
                if (entries[0].isIntersecting && hasMore && !loading && searchTerm === '') {
                    const nextPage = currentPage + 1;
                    setCurrentPage(nextPage);
                    dispatch(fetchAllHotels(nextPage, 10) as any);
                }
            },
            { threshold: 0.1 }
        );

        observerRef.current.observe(observerElement);
        return () => observerRef.current?.disconnect();
    }, [currentPage, hasMore, loading, searchTerm, dispatch]);

    // Update hasMore when pagination changes
    useEffect(() => {
        if (pagination) {
            setHasMore(pagination.hasNextPage);
        }
    }, [pagination]);

    // Handle search
    const handleSearch = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
        const term = e.target.value;
        setSearchTerm(term);

        if (term.trim() === '') {
            // Reset to all hotels
            dispatch(clearSearch());
            setCurrentPage(1);
            dispatch(fetchAllHotels(1, 10) as any);
        } else {
            // Search hotels
            dispatch(searchHotels(term, 1, 10) as any);
        }
    }, [dispatch]);

    const handleAddHotel = () => {
        navigate('/add-hotel');
    };

    const handleHotelClick = (hotelId: string) => {
        navigate(`/hotel/${hotelId}`);
    };

    // Display either search results or all hotels
    const displayHotels = searchTerm.trim() !== '' ? searchResults : hotels;
    const displayPagination = searchTerm.trim() !== '' ? searchPagination : pagination;

    const mockImageUrl = "https://picsum.photos/seed/hotel/400/300";

    return (
        <div className="min-h-screen bg-gray-50">
            <AdminNavbar role={role} />
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-8 gap-4">
                    <div>
                        <h1 className="text-3xl font-bold text-gray-900 bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-purple-600 inline-block">
                            All Hotels
                        </h1>
                        <p className="mt-1 text-sm text-gray-500">Manage and view all registered hotels</p>
                    </div>
                    
                    {role === 'admin' && (
                        <button
                            onClick={handleAddHotel}
                            className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transform hover:-translate-y-0.5 transition-all duration-200"
                        >
                            <span className="mr-2 text-lg">+</span> Add Hotel
                        </button>
                    )}
                </div>

                <div className="mb-8 relative max-w-2xl">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                        <span className="text-gray-400 text-lg">🔍</span>
                    </div>
                    <input
                        type="text"
                        className="block w-full pl-10 pr-3 py-3 border border-gray-300 rounded-xl leading-5 bg-white placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm shadow-sm transition-shadow duration-200 hover:shadow-md"
                        placeholder="Search by hotel name, city, or owner..."
                        value={searchTerm}
                        onChange={handleSearch}
                    />
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                    {displayHotels.length > 0 ? (
                        displayHotels.map((hotel) => (
                            <HotelCard
                                key={hotel.id}
                                name={hotel.name}
                                owner={hotel.ownerName}
                                city={hotel.city}
                                contact={hotel.contactNumber}
                                imageUrl={hotel.imageUrl || mockImageUrl}
                                onClick={() => handleHotelClick(hotel.id)}
                            />
                        ))
                    ) : (
                        <div className="col-span-full text-center py-12">
                            <p className="text-gray-500 text-lg">
                                {loading ? 'Loading hotels...' : 'No hotels found matching your search.'}
                            </p>
                        </div>
                    )}
                </div>

                {/* Infinite scroll trigger */}
                {searchTerm === '' && (
                    <div
                        ref={scrollContainerRef}
                        className="py-8 text-center"
                    >
                        {loading && <p className="text-indigo-600 font-medium animate-pulse">Loading more hotels...</p>}
                    </div>
                )}

                {/* Pagination info */}
                {displayPagination && (
                    <div className="text-center text-sm text-gray-500 mt-4 pb-8">
                        <p>
                            Showing {displayHotels.length} of {displayPagination.totalCount} hotels
                            {displayPagination.hasNextPage && ' (scroll to load more)'}
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default AllHotels;
