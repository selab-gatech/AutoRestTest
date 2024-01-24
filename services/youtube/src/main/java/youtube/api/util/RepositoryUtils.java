package youtube.api.util;

import youtube.api.model.*;

import java.security.SecureRandom;
import java.util.Arrays;
import java.util.Date;
import java.util.List;

public class RepositoryUtils {

    public static final List<String> topicIds = Arrays.asList("/m/04rlf","/m/02mscn","/m/0ggq0m","/m/01lyv","/m/02lkt","/m/0glt670","/m/05rwpb","/m/03_d0","/m/028sqc","/m/0g293","/m/064t9","/m/06cqb","/m/06j6l","/m/06by7","/m/0gywn","/m/0bzvm2","/m/025zzc","/m/02ntfj","/m/0b1vjn","/m/02hygl","/m/04q1x3q","/m/01sjng","/m/0403l3g","/m/021bp2","/m/022dc6","/m/03hf_rm","/m/06ntj","/m/0jm_","/m/018jz","/m/018w8","/m/01cgz","/m/09xp_","/m/02vx4","/m/037hz","/m/03tmr","/m/01h7lh","/m/0410tth","/m/07bs0","/m/07_53","/m/02jjt","/m/09kqc","/m/02vxn","/m/05qjc","/m/066wd","/m/0f2f9","/m/019_rr","/m/032tl","/m/027x7n","/m/02wbm","/m/03glg","/m/068hy","/m/041xxh","/m/07c1v","/m/07bxq","/m/07yv9","/m/098wr","/m/09s1f","/m/0kt51","/m/01h6rj","/m/05qt0","/m/06bvp","/m/01k8wb");

    private static final String CHAR_LOWER = "abcdefghijklmnopqrstuvwxyz";
    private static final String CHAR_UPPER = CHAR_LOWER.toUpperCase();
    private static final String NUMBER = "0123456789";

    private static final String DATA_FOR_RANDOM_STRING = CHAR_LOWER + CHAR_UPPER + NUMBER;
    private static SecureRandom random = new SecureRandom();

    private RepositoryUtils() {
        throw new UnsupportedOperationException("Trying to instance utility class");
    }

    public static SearchResult convertToSearchResult(Object resource, String part) {
        SearchResult result = new SearchResult();
        result.setEtag(generateEtag());
        result.setKind("youtube#searchResult");
        ResourceId resourceId = new ResourceId();

        if (resource instanceof Video) {
            resourceId.setKind("youtube#video");
            resourceId.setVideoId(((Video) resource).getId());
            if (!part.equals("id")) {
                result.setSnippet(((Video) resource).getSnippet());
            }
        } else if(resource instanceof Channel) {
            resourceId.setKind("youtube#channel");
            resourceId.setChannelId(((Channel) resource).getId());
            if(!part.equals("id")) {
                result.setSnippet(((Channel) resource).getSnippet());
            }
        } else if(resource instanceof Playlist) {
            resourceId.setKind("youtube#playlist");
            resourceId.setPlaylistId(((Playlist) resource).getId());
            if(!part.equals("id")) {
                result.setSnippet(((Playlist) resource).getSnippet());
            }
        }

        result.setId(resourceId);
        return result;
    }


    public static String generateEtag() {
        StringBuilder result = new StringBuilder(25);

        for(int i = 0; i < 25; i++) {
            int randomIndex = random.nextInt(DATA_FOR_RANDOM_STRING.length());
            char randomChar = DATA_FOR_RANDOM_STRING.charAt(randomIndex);
            result.append(randomChar);
        }

        return result.toString();
    }

    public static boolean filterByChannelId(Object x, String channelId) {
        boolean result = channelId == null || channelId.isEmpty();
        if(x instanceof Video) {
            Video v = (Video) x;
            result |= v.getSnippet().getChannelId().equals(channelId);
        } else if (x instanceof Channel) {
            Channel c = (Channel) x;
            result |= c.getSnippet().getChannelId().equals(channelId);
        } else if(x instanceof Playlist) {
            Playlist p = (Playlist) x;
            result |= p.getSnippet().getChannelId().equals(channelId);
        }
        return result;
    }

    public static boolean filterByDates(Object x, Date[] dates) {
        boolean result = dates[0] == null && dates[1] == null;
        if(x instanceof Video) {
            Video v = (Video) x;
            result |= ((dates[0] == null || v.getSnippet().getPublishedAt().before(dates[0])) && (dates[1] == null || v.getSnippet().getPublishedAt().after(dates[1])));
        } else if (x instanceof Channel) {
            Channel c = (Channel) x;
            result |= ((dates[0] == null || c.getSnippet().getPublishedAt().before(dates[0])) && (dates[1] == null || c.getSnippet().getPublishedAt().after(dates[1])));
        } else if(x instanceof Playlist) {
            Playlist p = (Playlist) x;
            result |= ((dates[0] == null || p.getSnippet().getPublishedAt().before(dates[0])) && (dates[1] == null || p.getSnippet().getPublishedAt().after(dates[1])));
        }
        return result;
    }

    public static boolean filterByQuery(Object x, String q) {
        boolean result = q == null || q.isEmpty();
        if(!result && !q.contains("-") && !q.contains("|") && x instanceof Video) {
            Video v = (Video) x;
            result = v.getSnippet().getTitle().toLowerCase().contains(q.toLowerCase());
        } else if (!result && !q.contains("-") && x instanceof Channel) {
            Channel c = (Channel) x;
            result = c.getSnippet().getTitle().toLowerCase().contains(q.toLowerCase());
        } else if(!result && !q.contains("-") && x instanceof Playlist) {
            Playlist p = (Playlist) x;
            result = p.getSnippet().getTitle().toLowerCase().contains(q.toLowerCase());
        } else if(!result && q.contains("|") && (!q.contains("-") || q.indexOf('|') < q.indexOf('-'))) {
            String[] sp = q.split("\\|");
            boolean exp1 = filterByQuery(x, sp[0].trim());
            boolean exp2 = filterByQuery(x, sp[1].trim());
            result = exp1 || exp2;
        } else if(!result && q.contains("-") && (!q.contains("|") || q.indexOf('|') > q.indexOf('-'))) {
            String[] sp = q.split("-");
            boolean exp1 = sp[0].trim().isEmpty() || filterByQuery(x, sp[0].trim());
            boolean exp2 = filterByQuery(x, sp[1].trim());
            result = exp1 && !exp2;
        }
        return result;
    }

    public static boolean filterByRelevanceLanguage(Object x, String relevanceLanguage) {
        boolean result = relevanceLanguage == null || relevanceLanguage.isEmpty();
        if(x instanceof Video) {
            Video v = (Video) x;
            result |= v.getDefaultLanguage().equalsIgnoreCase(relevanceLanguage);
        } else if (x instanceof Channel) {
            Channel c = (Channel) x;
            result |= c.getDefaultLanguage().equalsIgnoreCase(relevanceLanguage);
        } else if(x instanceof Playlist) {
            Playlist p = (Playlist) x;
            result |= p.getDefaultLanguage().equalsIgnoreCase(relevanceLanguage);
        }
        return result;
    }

    public static boolean filterByTopicId(Object x, String topicId) {
        boolean result = topicId == null || topicId.isEmpty() || !topicIds.contains(topicId);
        if(x instanceof Video) {
            Video v = (Video) x;
            result |= v.getRelevantTopicIds() != null && v.getRelevantTopicIds().contains(topicId);
        } else if (x instanceof Channel) {
            Channel c = (Channel) x;
            result |= c.getRelevantTopicIds() != null && c.getRelevantTopicIds().contains(topicId);
        } else if(x instanceof Playlist) {
            Playlist p = (Playlist) x;
            result |= p.getVideos().stream().anyMatch(v -> v.getRelevantTopicIds() != null && v.getRelevantTopicIds().contains(topicId));
        }
        return result;
    }

}
