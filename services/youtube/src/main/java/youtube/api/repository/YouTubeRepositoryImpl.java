package youtube.api.repository;

import youtube.api.model.*;
import youtube.api.util.BadPaginationException;
import youtube.api.util.Location;
import youtube.api.util.RepositoryUtils;

import javax.ws.rs.BadRequestException;
import javax.ws.rs.InternalServerErrorException;
import javax.ws.rs.NotFoundException;
import javax.ws.rs.core.Response;
import java.text.ParseException;
import java.util.*;

public class YouTubeRepositoryImpl implements YouTubeRepository {

    YouTubeVideoRepository videoRepository;
    YoutubeChannelRepository channelRepository;
    YoutubePlaylistRepository playlistRepository;
    private static YouTubeRepositoryImpl instance;

    private static final String BAD_REQUEST_INVALID_CHANNEL_ID = "Invalid channel.";
    private static final String BAD_REQUEST_INVALID_RELATED_TO_VIDEO = "Invalid video.";
    private static final String NOT_FOUND_CONTENT_OWNER = "The specified content owner account was not found.";
    private static final String VISITOR_ID = RepositoryUtils.generateEtag();

    private YouTubeRepositoryImpl() {}

    public static YouTubeRepositoryImpl getInstance() throws ParseException {
        if(instance == null) {
            instance = new YouTubeRepositoryImpl();
            instance.init();
        }
        return instance;
    }

    private void init() throws ParseException {
        videoRepository = YouTubeVideoRepository.getInstance();
        channelRepository = YoutubeChannelRepository.getInstance();
        playlistRepository = YoutubePlaylistRepository.getInstance(videoRepository.getVideos());
    }


    @Override
    public SearchListResponse searchList(SearchRequest searchRequest, String partType, Location location, Date[] dates, byte typeSearch, Locale locale) throws BadPaginationException {

        if(searchRequest.getChannelId() != null && this.channelRepository.getChannelById(searchRequest.getChannelId()) == null) {
            throw new BadRequestException(BAD_REQUEST_INVALID_CHANNEL_ID, Response.status(400).entity(generateErrorResponse(BAD_REQUEST_INVALID_CHANNEL_ID, "channelId", 400)).build());
        }

        if(searchRequest.getOnBehalfOfContentOwner() != null && !this.channelRepository.existsContentOwner(searchRequest.getOnBehalfOfContentOwner())) {
            throw new NotFoundException(NOT_FOUND_CONTENT_OWNER, Response.status(404).entity(generateErrorResponse(NOT_FOUND_CONTENT_OWNER, "onBehalfOfContentOwner", 404)).build());
        }

        SearchListResponse response = new SearchListResponse();
        List<SearchResult> items;
        if(searchRequest.getForContentOwner() != null && searchRequest.getForContentOwner()) {
            items = searchForContentOwnerFilter(searchRequest, partType, location, dates, locale);
        } else if(searchRequest.getForMine() != null && searchRequest.getForMine()) {
            items = searchForMineFilter(searchRequest, partType, location, dates, locale);
        } else if(searchRequest.getRelatedToVideoId() != null && !searchRequest.getRelatedToVideoId().isEmpty()) {
            items = searchRelatedToVideoFilter(searchRequest, partType, locale);
        } else {
            items = defaultSearch(searchRequest, partType, location, dates, typeSearch, locale);
        }

        response.setTokenPagination(new TokenPagination());
        response.setEtag("'abcdefghijk_lmnopqrstuvwxyz1234567890'");
        response.setKind("youtube#searchListResponse");
        response.setRegionCode(locale.getCountry());
        response.setVisitorId(VISITOR_ID);
        response.setEventId(RepositoryUtils.generateEtag());

        sort(items, searchRequest.getOrder());
        pagination(response, items, searchRequest.getMaxResults(), searchRequest.getPageToken());
        return response;
    }

    private List<SearchResult> searchForContentOwnerFilter(SearchRequest searchRequest, String partType, Location location, Date[] dates, Locale locale) {
        return this.videoRepository.processForContentOwnerRequest(searchRequest, partType, location, dates, locale);
    }

    private List<SearchResult> searchForMineFilter(SearchRequest searchRequest, String partType, Location location, Date[] dates, Locale locale) {
        return this.videoRepository.processForMineRequest(searchRequest, partType, location, dates, locale);
    }

    private List<SearchResult> searchRelatedToVideoFilter(SearchRequest searchRequest, String partType, Locale locale) {
        Video v = this.videoRepository.getVideoById(searchRequest.getRelatedToVideoId());
        if(v == null && searchRequest.getRelatedToVideoId() != null) {
            throw new BadRequestException(BAD_REQUEST_INVALID_RELATED_TO_VIDEO, Response.status(400).entity(generateErrorResponse(BAD_REQUEST_INVALID_RELATED_TO_VIDEO, "relatedToVideo", 400)).build());
        }
        return this.videoRepository.processRelatedToVideoRequest(searchRequest, v, partType, locale);
    }

    private List<SearchResult> defaultSearch(SearchRequest searchRequest, String partType, Location location, Date[] dates, byte typeSearch, Locale locale) {
        List<SearchResult> items = new ArrayList<>();
        //This if statement checks if the LSB is set or the byte is 0. If so, the API will search for videos.
        if(typeSearch == 0 || (typeSearch & 1) != 0) {
            items.addAll(this.videoRepository.processDefaultRequest(searchRequest, partType, location, dates, locale));
        }
        //This if statement checks if the second LSB is set or the byte is 0. If so, the API will search for playlists.
        if((typeSearch & 0xff & (1<<1)) != 0) {
            items.addAll(this.playlistRepository.processDefaultRequest(searchRequest, partType, dates));
        }
        //This if statement checks if the MSB is set or the byte is 0. If so, the API will search for channels.
        if((typeSearch & 0xff & (1<<2)) != 0) {
            items.addAll(this.channelRepository.processDefaultRequest(searchRequest, partType, dates));
        }

        return items;
    }

    private void sort(List<SearchResult> items, String order) {
        if(order != null && !order.isEmpty() && order.equals("date")) {
            items.sort(Comparator.comparing((SearchResult x) -> x.getSnippet().getPublishedAt()).reversed());
        } else if(order != null && !order.isEmpty() && order.equals("title")) {
            items.sort(Comparator.comparing(x -> x.getSnippet().getTitle()));
        } else if(order != null && !order.isEmpty() && order.equals("viewCount")) {
            items.sort(new ViewCountSort(videoRepository, channelRepository, playlistRepository));
        }
    }

    private void pagination(SearchListResponse response, List<SearchResult> items, Integer maxResults, String pageToken) throws BadPaginationException {
        int pages = (int) Math.ceil((double) items.size()/(double) maxResults);
        int page;
        try {
            page = pageToken == null || pageToken.isEmpty()? 1 : Integer.parseInt(pageToken.split("-")[1]);
        } catch (IndexOutOfBoundsException | NumberFormatException e) {
            throw new BadPaginationException();
        }

        int totalResults = items.size();

        if(page < pages) {
            items = items.subList(maxResults*(page-1), maxResults*page);
        } else if(page == pages) {
            items = items.subList(maxResults*(page-1), totalResults);
        } else if(pages != 0) {
            throw new BadPaginationException();
        }

        if(page != 1) {
            response.setPrevPageToken("page-" + (page-1));
        }
        if(pages != 0 && page != pages) {
            response.setNextPageToken("page-" + (page+1));
        }

        PageInfo pageInfo = new PageInfo();
        pageInfo.setResultsPerPage(maxResults);
        pageInfo.setTotalResults(totalResults);
        response.setPageInfo(pageInfo);
        response.setItems(items);
    }

    private ErrorResponse generateErrorResponse(String message, String parameter, int statusCode) {
        ErrorResponse response = new ErrorResponse();
        response.setStatus(statusCode);
        response.setMessage(message);

        ErrorDescription description = new ErrorDescription();
        description.setMessage(message);
        description.setLocationType("parameter");
        description.setLocation(parameter);

        if(statusCode == 404) {
            description.setDomain("youtube.parameter");
            description.setReason("contentOwnerAccountNotFound");

        } else if(statusCode == 400 && parameter.equals("relatedToVideo")) {
            description.setDomain("youtube.search");
            description.setReason("invalidChannelId");
        } else if(statusCode == 400 && parameter.equals("channelId")) {
            description.setDomain("youtube.search");
            description.setReason("invalidVideoId");
        }

        response.setErrors(Collections.singletonList(description));
        return response;
    }

    public static class ViewCountSort implements Comparator<SearchResult> {

        private YouTubeVideoRepository videoRepository;
        private YoutubeChannelRepository channelRepository;
        private YoutubePlaylistRepository playlistRepository;

        public ViewCountSort(YouTubeVideoRepository videoRepository, YoutubeChannelRepository channelRepository, YoutubePlaylistRepository playlistRepository) {
            this.videoRepository = videoRepository;
            this.channelRepository = channelRepository;
            this.playlistRepository = playlistRepository;
        }

        @Override
        public int compare(SearchResult o1, SearchResult o2) {
            Video v1 = o1.getId().getKind().equals("youtube#video")? this.videoRepository.getVideoById(o1.getId().getVideoId()) : null;
            Video v2 = o2.getId().getKind().equals("youtube#video")? this.videoRepository.getVideoById(o2.getId().getVideoId()) : null;
            Channel c1 = o1.getId().getKind().equals("youtube#channel")? this.channelRepository.getChannelById(o1.getId().getChannelId()) : null;
            Channel c2 = o2.getId().getKind().equals("youtube#channel")? this.channelRepository.getChannelById(o2.getId().getChannelId()) : null;
            Playlist p1 = o1.getId().getKind().equals("youtube#playlist")? this.playlistRepository.getPlaylistById(o1.getId().getPlaylistId()) : null;
            Playlist p2 = o2.getId().getKind().equals("youtube#playlist")? this.playlistRepository.getPlaylistById(o2.getId().getPlaylistId()) : null;

            Long viewCount1;
            Long viewCount2;

            viewCount1 = getViewCount(v1, c1, p1);

            viewCount2 = getViewCount(v2, c2, p2);

            return viewCount2.compareTo(viewCount1);
        }

        private Long getViewCount(Video v, Channel c, Playlist p) {
            Long viewCount;
            if(v != null) {
                viewCount = v.getHasLiveStreamingDetails() != null && v.getHasLiveStreamingDetails()? v.getLiveStreamingConcurrentViewers() : v.getViewCount();
            } else if(c != null) {
                viewCount = c.getViewCount();
            } else if(p != null){
                viewCount = (long) p.getVideos().stream().mapToLong(Video::getViewCount).average().orElse(0);
            } else {
                throw new InternalServerErrorException();
            }
            return viewCount;
        }
    }

}
