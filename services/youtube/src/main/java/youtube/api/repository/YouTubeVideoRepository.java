package youtube.api.repository;

import youtube.api.model.*;
import youtube.api.util.Location;

import java.text.ParseException;
import java.util.*;
import java.util.stream.Collectors;

import static youtube.data.VideoData.getVideoData;
import static youtube.api.util.RepositoryUtils.*;

public class YouTubeVideoRepository {

    private static YouTubeVideoRepository instance = null;
    private YoutubeChannelRepository channelRepository;
    private List<Video> videos;

    private YouTubeVideoRepository() throws ParseException {
        videos = instantiate();
        channelRepository = YoutubeChannelRepository.getInstance();
    }

    public static YouTubeVideoRepository getInstance() throws ParseException {
        if(instance == null) {
            instance = new YouTubeVideoRepository();
        }

        return instance;
    }

    private static List<Video> instantiate() throws ParseException {
        return getVideoData();
    }

    public List<SearchResult> processForContentOwnerRequest(SearchRequest searchRequest, String partType, Location location, Date[] dates, Locale locale) {
        return videos.stream()
                .filter(x -> filterByOnBehalfOfContentOwner(x, searchRequest.getForContentOwner(), searchRequest.getOnBehalfOfContentOwner()))
                .filter(x -> filterByChannelId(x, searchRequest.getChannelId()))
                .filter(x -> filterByEventType(x, searchRequest.getEventType()))
                .filter(x -> filterByLocation(x, location))
                .filter(x -> filterByDates(x, dates))
                .filter(x -> filterByQuery(x, searchRequest.getQ()))
                .filter(x -> filterByRelevanceLanguage(x, searchRequest.getRelevanceLanguage()))
                .filter(x -> filterByRegionCode(x, searchRequest.getRegionCode()))
                .filter(x -> filterBySafeSearch(x, searchRequest.getSafeSearch(), locale))
                .filter(x -> filterByTopicId(x, searchRequest.getTopicId()))
                .filter(x -> filterByVideoCaption(x, searchRequest.getVideoCaption()))
                .filter(x -> filterByVideoCategoryId(x, searchRequest.getVideoCategoryId()))
                .map(x -> convertToSearchResult(x, partType))
                .collect(Collectors.toList());
    }

    public List<SearchResult> processForMineRequest(SearchRequest searchRequest, String partType, Location location, Date[] dates, Locale locale) {
        return videos.stream()
                .filter(x -> filterForMine(x, searchRequest.getForMine()))
                .filter(x -> filterByChannelId(x, searchRequest.getChannelId()))
                .filter(x -> filterByEventType(x, searchRequest.getEventType()))
                .filter(x -> filterByLocation(x, location))
                .filter(x -> filterByDates(x, dates))
                .filter(x -> filterByQuery(x, searchRequest.getQ()))
                .filter(x -> filterByRelevanceLanguage(x, searchRequest.getRelevanceLanguage()))
                .filter(x -> filterByRegionCode(x, searchRequest.getRegionCode()))
                .filter(x -> filterBySafeSearch(x, searchRequest.getSafeSearch(), locale))
                .filter(x -> filterByTopicId(x, searchRequest.getTopicId()))
                .filter(x -> filterByVideoCaption(x, searchRequest.getVideoCaption()))
                .filter(x -> filterByVideoCategoryId(x, searchRequest.getVideoCategoryId()))
                .map(x -> convertToSearchResult(x, partType))
                .collect(Collectors.toList());
    }

    public List<SearchResult> processRelatedToVideoRequest(SearchRequest searchRequest, Video video, String partType, Locale locale) {
        return videos.stream()
                .filter(x -> filterByRelatedToVideo(x, video))
                .filter(x -> filterByRegionCode(x, searchRequest.getRegionCode()))
                .filter(x -> filterByRelevanceLanguage(x, searchRequest.getRelevanceLanguage()))
                .filter(x -> filterBySafeSearch(x, searchRequest.getSafeSearch(), locale))
                .map(x -> convertToSearchResult(x, partType))
                .collect(Collectors.toList());
    }

    public List<SearchResult> processDefaultRequest(SearchRequest searchRequest, String partType, Location location, Date[] dates, Locale locale) {
        return videos.stream()
                .filter(x -> filterForDeveloper(x, searchRequest.getForDeveloper()))
                .filter(x -> filterByChannelId(x, searchRequest.getChannelId()))
                .filter(x -> filterByEventType(x, searchRequest.getEventType()))
                .filter(x -> filterByLocation(x, location))
                .filter(x -> filterByDates(x, dates))
                .filter(x -> filterByQuery(x, searchRequest.getQ()))
                .filter(x -> filterByRelevanceLanguage(x, searchRequest.getRelevanceLanguage()))
                .filter(x -> filterByRegionCode(x, searchRequest.getRegionCode()))
                .filter(x -> filterBySafeSearch(x, searchRequest.getSafeSearch(), locale))
                .filter(x -> filterByTopicId(x, searchRequest.getTopicId()))
                .filter(x -> filterByVideoCaption(x, searchRequest.getVideoCaption()))
                .filter(x -> filterByVideoCategoryId(x, searchRequest.getVideoCategoryId()))
                .filter(x -> filterByVideoDefinition(x, searchRequest.getVideoDefinition()))
                .filter(x -> filterByVideoDimension(x, searchRequest.getVideoDimension()))
                .filter(x -> filterByVideoDuration(x, searchRequest.getVideoDuration()))
                .filter(x -> filterByVideoEmbeddable(x, searchRequest.getVideoEmbeddable()))
                .filter(x -> filterByVideoLicense(x, searchRequest.getVideoLicense()))
                .filter(x -> filterByVideoType(x, searchRequest.getVideoType()))
                .map(x -> convertToSearchResult(x, partType))
                .collect(Collectors.toList());
    }

    public Video getVideoById(String videoId) {
        return videos.stream()
                .filter(x -> x.getId().equals(videoId))
                .findFirst()
                .orElse(null);
    }

    private boolean filterByOnBehalfOfContentOwner(Video x, Boolean forContentOwner, String onBehalfOfContentOwner) {
        boolean result = !forContentOwner;
        if(!result) {
            List<String> channelIds = this.channelRepository.getChannels().stream()
                    .filter(c -> c.getContentOwnerId().equals(onBehalfOfContentOwner))
                    .map(Channel::getId)
                    .collect(Collectors.toList());
            result = channelIds.contains(x.getSnippet().getChannelId());
        }
        return result;
    }

    private boolean filterForMine(Video x, Boolean forMine) {
        return !forMine || x.getSnippet().getChannelId().equals("UCxwvutsrqponmlkjihgfedc");
    }

    private boolean filterByRelatedToVideo(Video x, Video videoRelated) {
        return videoRelated == null ||
                (x.getRelevantTopicIds() != null && x.getRelevantTopicIds().stream().anyMatch(y -> videoRelated.getRelevantTopicIds() != null && videoRelated.getRelevantTopicIds().contains(y)))
                || x.getCategory().getId().equals(videoRelated.getCategory().getId());
    }

    private boolean filterForDeveloper(Video x, Boolean forDeveloper) {
        return forDeveloper == null || !forDeveloper || x.getForDeveloperVideo();
    }

    private boolean filterByEventType(Video x, String eventType) {
        boolean result = (eventType == null || eventType.isEmpty());
        if(!result) {
            switch (eventType) {
                case "completed":
                    result = x.getHasLiveStreamingDetails() && x.getSnippet().getLiveBroadcastContent().equals(SearchResultSnippet.LiveBroadcastContent.NONE);
                    break;
                case "live":
                    result = x.getHasLiveStreamingDetails() && x.getSnippet().getLiveBroadcastContent().equals(SearchResultSnippet.LiveBroadcastContent.LIVE);
                    break;
                case "upcoming":
                    result = x.getHasLiveStreamingDetails() && x.getSnippet().getLiveBroadcastContent().equals(SearchResultSnippet.LiveBroadcastContent.UPCOMING);
                    break;
                default:
                    result = false;
            }
        }
        return result;
    }

    /**
     * This method implements the haversine formula to calculate the distance between two different coordinates. Then, the method checks if
     * the distance is minor than the radius specified.
     * @param x The video with the data of the location
     * @param location The location and the radius specified by the user
     * @return a boolean that indicates if the location of the video is in the requested radius
     */
    private boolean filterByLocation(Video x, Location location) {
        boolean result = location.getLongitude() == null && location.getLatitude() == null;
        if(!result && x.getLongitude() != null && x.getLatitude() != null) {

            double earthRadius = 6371e3;
            double lat1 = Math.toRadians(x.getLatitude());
            double lat2 = Math.toRadians(location.getLatitude());
            double dlat = lat2-lat1;
            double dlon = Math.toRadians(location.getLongitude()-x.getLongitude());

            double haversine = Math.pow(Math.sin(dlat/2), 2) + Math.cos(lat1) * Math.cos(lat2) * Math.pow(Math.sin(dlon/2), 2);
            double c = 2 * Math.atan2(Math.sqrt(haversine), Math.sqrt(1-haversine));

            double distance = earthRadius * c;

            result = distance < location.getRadius();
        }

        return result;
    }

    private boolean filterByRegionCode(Video x, String regionCode) {
        return (regionCode == null || regionCode.isEmpty()) || ((x.getContentDetailsRegionRestrictionAllowed() == null || x.getContentDetailsRegionRestrictionAllowed().contains(regionCode.toLowerCase())) &&
                (x.getContentDetailsRegionRestrictionBlocked() == null || !x.getContentDetailsRegionRestrictionBlocked().contains(regionCode)));
    }

    private boolean filterBySafeSearch(Video x, String safeSearch, Locale locale) {
        boolean result;
        if(safeSearch == null || safeSearch.isEmpty()) {
            safeSearch = "moderate";
        }
        switch (safeSearch) {
            case "none":
                result = true;
                break;
            case "moderate":
                result = filterByRegionCode(x, locale.getCountry());
                break;
            case "strict":
                result = filterByRegionCode(x, locale.getCountry()) && !"ytAgeRestricted".equals(x.getYtRating());
                break;
            default:
                result = false;
        }
        return result;
    }

    private boolean filterByVideoCaption(Video x, String videoCaption) {
        boolean result = videoCaption == null || videoCaption.isEmpty() || videoCaption.equals("any");
        if(!result) {
            if(videoCaption.equals("closedCaption")) {
                result = x.getContentDetailsCaption();
            } else if(videoCaption.equals("none")) {
                result = !x.getContentDetailsCaption();
            }
        }
        return result;
    }

    private boolean filterByVideoCategoryId(Video x, String videoCategoryId) {
        return (videoCategoryId == null || videoCategoryId.isEmpty()) || (x.getCategory() != null && x.getCategory().getId().equals(videoCategoryId));
    }

    private boolean filterByVideoDefinition(Video x, String videoDefinition) {
        return (videoDefinition == null || videoDefinition.isEmpty()) || videoDefinition.equals("any") || x.getContentDetailsDefinition().getDefinition().equals(videoDefinition);
    }

    private boolean filterByVideoDimension(Video x, String videoDimension) {
        return (videoDimension == null || videoDimension.isEmpty()) || videoDimension.equals("any") || x.getContentDetailsDimension().getDimension().equals(videoDimension);
    }

    private boolean filterByVideoDuration(Video x, String videoDuration) {
        boolean result = videoDuration == null || videoDuration.isEmpty() || videoDuration.equals("any");
        if(!result) {
            switch (videoDuration) {
                case "long":
                    result = x.getContentDetailsDuration().getSeconds() > 1200;
                    break;
                case "medium":
                    result = x.getContentDetailsDuration().getSeconds() >= 240 && x.getContentDetailsDuration().getSeconds() <= 1200;
                    break;
                case "short":
                    result = x.getContentDetailsDuration().getSeconds() < 240;
                    break;
                default:
                    result = false;
            }
        }
        return result;
    }

    private boolean filterByVideoEmbeddable(Video x, String videoEmbeddable) {
        return !Boolean.parseBoolean(videoEmbeddable) || x.getEmbeddable();
    }

    private boolean filterByVideoLicense(Video x, String videoLicense) {
        return (videoLicense == null || videoLicense.isEmpty()) || videoLicense.equals("any") || x.getLicense().getLicense().equals(videoLicense);
    }

    private boolean filterByVideoType(Video x, String videoType) {
        return (videoType == null || videoType.isEmpty()) || videoType.equals("any") || x.getType().equals(videoType);
    }

    protected List<Video> getVideos() {
        return this.videos;
    }
}
