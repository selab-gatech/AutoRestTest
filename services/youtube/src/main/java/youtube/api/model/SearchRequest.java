package youtube.api.model;

import javax.ws.rs.QueryParam;

public class SearchRequest {

    @QueryParam("part")
    private String part;
    @QueryParam("forContentOwner")
    private Boolean forContentOwner;
    @QueryParam("forDeveloper")
    private Boolean forDeveloper;
    @QueryParam("forMine")
    private Boolean forMine;
    @QueryParam("relatedToVideoId")
    private String relatedToVideoId;
    @QueryParam("channelId")
    private String channelId;
    @QueryParam("channelType")
    private String channelType;
    @QueryParam("eventType")
    private String eventType;
    @QueryParam("location")
    private String location;
    @QueryParam("locationRadius")
    private String locationRadius;
    @QueryParam("maxResults")
    private Integer maxResults;
    @QueryParam("onBehalfOfContentOwner")
    private String onBehalfOfContentOwner;
    @QueryParam("order")
    private String order;
    @QueryParam("pageToken")
    private String pageToken;
    @QueryParam("publishedAfter")
    private String publishedAfterStr;
    @QueryParam("publishedBefore")
    private String publishedBeforeStr;
    @QueryParam("q")
    private String q;
    @QueryParam("relevanceLanguage")
    private String relevanceLanguage;
    @QueryParam("regionCode")
    private String regionCode;
    @QueryParam("safeSearch")
    private String safeSearch;
    @QueryParam("topicId")
    private String topicId;
    @QueryParam("type")
    private String type;
    @QueryParam("videoCaption")
    private String videoCaption;
    @QueryParam("videoCategoryId")
    private String videoCategoryId;
    @QueryParam("videoDefinition")
    private String videoDefinition;
    @QueryParam("videoDimension")
    private String videoDimension;
    @QueryParam("videoDuration")
    private String videoDuration;
    @QueryParam("videoEmbeddable")
    private String videoEmbeddable;
    @QueryParam("videoLicense")
    private String videoLicense;
    @QueryParam("videoSyndicated")
    private String videoSyndicated;
    @QueryParam("videoType")
    private String videoType;

    public String getPart() {
        return part;
    }

    public void setPart(String part) {
        this.part = part;
    }

    public Boolean getForContentOwner() {
        return forContentOwner;
    }

    public void setForContentOwner(Boolean forContentOwner) {
        this.forContentOwner = forContentOwner;
    }

    public Boolean getForDeveloper() {
        return forDeveloper;
    }

    public void setForDeveloper(Boolean forDeveloper) {
        this.forDeveloper = forDeveloper;
    }

    public Boolean getForMine() {
        return forMine;
    }

    public void setForMine(Boolean forMine) {
        this.forMine = forMine;
    }

    public String getRelatedToVideoId() {
        return relatedToVideoId;
    }

    public void setRelatedToVideoId(String relatedToVideoId) {
        this.relatedToVideoId = relatedToVideoId;
    }

    public String getChannelId() {
        return channelId;
    }

    public void setChannelId(String channelId) {
        this.channelId = channelId;
    }

    public String getChannelType() {
        return channelType;
    }

    public void setChannelType(String channelType) {
        this.channelType = channelType;
    }

    public String getEventType() {
        return eventType;
    }

    public void setEventType(String eventType) {
        this.eventType = eventType;
    }

    public String getLocation() {
        return location;
    }

    public void setLocation(String location) {
        this.location = location;
    }

    public String getLocationRadius() {
        return locationRadius;
    }

    public void setLocationRadius(String locationRadius) {
        this.locationRadius = locationRadius;
    }

    public Integer getMaxResults() {
        return maxResults;
    }

    public void setMaxResults(Integer maxResults) {
        this.maxResults = maxResults;
    }

    public String getOnBehalfOfContentOwner() {
        return onBehalfOfContentOwner;
    }

    public void setOnBehalfOfContentOwner(String onBehalfOfContentOwner) {
        this.onBehalfOfContentOwner = onBehalfOfContentOwner;
    }

    public String getOrder() {
        return order;
    }

    public void setOrder(String order) {
        this.order = order;
    }

    public String getPageToken() {
        return pageToken;
    }

    public void setPageToken(String pageToken) {
        this.pageToken = pageToken;
    }

    public String getPublishedAfterStr() {
        return publishedAfterStr;
    }

    public void setPublishedAfterStr(String publishedAfterStr) {
        this.publishedAfterStr = publishedAfterStr;
    }

    public String getPublishedBeforeStr() {
        return publishedBeforeStr;
    }

    public void setPublishedBeforeStr(String publishedBeforeStr) {
        this.publishedBeforeStr = publishedBeforeStr;
    }

    public String getQ() {
        return q;
    }

    public void setQ(String q) {
        this.q = q;
    }

    public String getRelevanceLanguage() {
        return relevanceLanguage;
    }

    public void setRelevanceLanguage(String relevanceLanguage) {
        this.relevanceLanguage = relevanceLanguage;
    }

    public String getRegionCode() {
        return regionCode;
    }

    public void setRegionCode(String regionCode) {
        this.regionCode = regionCode;
    }

    public String getSafeSearch() {
        return safeSearch;
    }

    public void setSafeSearch(String safeSearch) {
        this.safeSearch = safeSearch;
    }

    public String getTopicId() {
        return topicId;
    }

    public void setTopicId(String topicId) {
        this.topicId = topicId;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public String getVideoCaption() {
        return videoCaption;
    }

    public void setVideoCaption(String videoCaption) {
        this.videoCaption = videoCaption;
    }

    public String getVideoCategoryId() {
        return videoCategoryId;
    }

    public void setVideoCategoryId(String videoCategoryId) {
        this.videoCategoryId = videoCategoryId;
    }

    public String getVideoDefinition() {
        return videoDefinition;
    }

    public void setVideoDefinition(String videoDefinition) {
        this.videoDefinition = videoDefinition;
    }

    public String getVideoDimension() {
        return videoDimension;
    }

    public void setVideoDimension(String videoDimension) {
        this.videoDimension = videoDimension;
    }

    public String getVideoDuration() {
        return videoDuration;
    }

    public void setVideoDuration(String videoDuration) {
        this.videoDuration = videoDuration;
    }

    public String getVideoEmbeddable() {
        return videoEmbeddable;
    }

    public void setVideoEmbeddable(String videoEmbeddable) {
        this.videoEmbeddable = videoEmbeddable;
    }

    public String getVideoLicense() {
        return videoLicense;
    }

    public void setVideoLicense(String videoLicense) {
        this.videoLicense = videoLicense;
    }

    public String getVideoSyndicated() {
        return videoSyndicated;
    }

    public void setVideoSyndicated(String videoSyndicated) {
        this.videoSyndicated = videoSyndicated;
    }

    public String getVideoType() {
        return videoType;
    }

    public void setVideoType(String videoType) {
        this.videoType = videoType;
    }
}
