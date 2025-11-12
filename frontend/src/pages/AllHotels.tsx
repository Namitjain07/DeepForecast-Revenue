import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import AdminNavbar from '../components/dashboard/AdminNavbar';
import HotelCard from '../components/dashboard/HotelCard';
import '../stylesheet/pages/page-all-hotel.css'
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
        <div>
            <AdminNavbar role={role} />
            <div className="page-all-hotel">
                <div className="page-all-hotel-header">
                    <div className="page-all-hotel-header-top">
                        <h1>All Hotels</h1>
                        {role === 'admin' && (
                            <div
                                className='page-all-hotel-add-button'
                                onClick={handleAddHotel}
                            >
                                + Add Hotel
                            </div>
                        )}
                    </div>

                    <div className="page-all-hotel-search-container">
                        <input
                            type="text"
                            className="page-all-hotel-search-bar"
                            placeholder="Search by hotel name, city, or owner..."
                            value={searchTerm}
                            onChange={handleSearch}
                        />
                    </div>
                </div>

                <div className="page-all-hotel-grid">
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
                        <p className="page-all-hotel-no-results">
                            {loading ? 'Loading hotels...' : 'No hotels found matching your search.'}
                        </p>
                    )}
                </div>

                {/* Infinite scroll trigger */}
                {searchTerm === '' && (
                    <div
                        ref={scrollContainerRef}
                        className="page-all-hotel-infinite-scroll-trigger"
                    >
                        {loading && <p className="page-all-hotel-loading">Loading more hotels...</p>}
                    </div>
                )}

                {/* Pagination info */}
                {displayPagination && (
                    <div className="page-all-hotel-pagination-info">
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
