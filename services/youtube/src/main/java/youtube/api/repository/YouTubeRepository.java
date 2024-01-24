package youtube.api.repository;

import youtube.api.model.SearchListResponse;
import youtube.api.model.SearchRequest;
import youtube.api.util.BadPaginationException;
import youtube.api.util.Location;

import java.util.Date;
import java.util.Locale;

public interface YouTubeRepository {

    SearchListResponse searchList(SearchRequest searchRequest, String partType, Location location, Date[] dates, byte typeSearch, Locale locale) throws BadPaginationException;
}
