package youtube.api.repository;

import youtube.api.model.*;
import youtube.api.util.RepositoryUtils;

import java.text.ParseException;
import java.util.*;
import java.util.stream.Collectors;

import static youtube.data.ChannelData.getChannelData;
import static youtube.api.util.RepositoryUtils.*;

public class YoutubeChannelRepository {

    private static YoutubeChannelRepository instance = null;
    private List<Channel> channels;

    private YoutubeChannelRepository() throws ParseException {
        this.channels = instantiate();
    }

    public static YoutubeChannelRepository getInstance() throws ParseException {
        if(instance == null) {
            instance = new YoutubeChannelRepository();
        }

        return instance;
    }

    private List<Channel> instantiate() throws ParseException {
        return getChannelData();
    }

    public List<SearchResult> processDefaultRequest(SearchRequest searchRequest, String partType, Date[] dates) {
        List<SearchResult> result = new ArrayList<>();
        if(searchRequest.getForDeveloper() == null || !searchRequest.getForDeveloper()) {
            result = channels.stream()
                    .filter(x -> filterByChannelId(x, searchRequest.getChannelId()))
                    .filter(x -> filterByChannelType(x, searchRequest.getChannelType()))
                    .filter(x -> filterByDates(x, dates))
                    .filter(x -> filterByQuery(x, searchRequest.getQ()))
                    .filter(x -> filterByRelevanceLanguage(x, searchRequest.getRelevanceLanguage()))
                    .filter(x -> filterByTopicId(x, searchRequest.getTopicId()))
                    .map(x -> convertToSearchResult(x, partType))
                    .collect(Collectors.toList());
        }
        return result;
    }

    protected List<Channel> getChannels() {
        return channels;
    }

    public boolean existsContentOwner(String contentOwnerId) {
        return channels.stream().anyMatch(x -> x.getContentOwnerId().equals(contentOwnerId));
    }

    public Channel getChannelById(String channelId) {
        return channels.stream()
                .filter(x -> x.getId().equals(channelId))
                .findFirst()
                .orElse(null);
    }

    private boolean filterByChannelType(Channel x, String channelType) {
        return (channelType == null || channelType.isEmpty()) || channelType.equals("any") || x.getType().equals(channelType);
    }
}
